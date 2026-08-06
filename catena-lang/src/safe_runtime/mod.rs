//! Process-isolated execution for Catena programs.
//!
//! Generated GPU code runs through native libraries, so a failed assertion,
//! GPU runtime failure, or native crash can terminate the process. [`SafeRuntime`]
//! executes [`Runtime`] in a child and transports device memory with HIP/CUDA IPC.
//! All IPC allocation and copying lives in this module; Runtime is used only
//! through its public API.

use std::{
    collections::HashMap,
    env, fs,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};

use thiserror::Error;

mod protocol;
mod transport;

pub use self::transport::SafeMemoryError;
use self::{
    protocol::{
        ProtocolError, RemoteExecError, Request, Response, WireCapability, WireGpuDialect,
        WireMemory, WireValue, read_frame, write_frame,
    },
    transport::{GpuTransport, IpcHandle, IpcMapping, OwnedAllocation},
};
use crate::{
    codegen::GpuDialect,
    runtime::{ExecError, MemOwn, MemRef, Runtime, Value},
};

const CHILD_MODE_ENV: &str = "CATENA_SAFE_RUNTIME_CHILD";

/// Initialization failures for [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeInitError {
    #[error("failed to identify the current executable: {0}")]
    CurrentExecutable(#[source] io::Error),
    #[error("failed to read Catena source {path}: {source}")]
    ReadSource {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to spawn SafeRuntime child {executable}: {source}")]
    Spawn {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("SafeRuntime initialization transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child initialization failed: {0}")]
    RemoteInitialization(String),
    #[error("SafeRuntime child returned an unexpected initialization response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated during initialization with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] SafeMemoryError),
}

/// Execution failures reported by [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeExecError {
    #[error(transparent)]
    Runtime(#[from] ExecError),
    #[error(transparent)]
    Memory(#[from] SafeMemoryError),
    #[error("SafeRuntime child memory operation failed: {0}")]
    RemoteMemory(String),
    #[error("SafeRuntime transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child returned an unexpected execution response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
}

/// Failure in the worker-mode entrypoint itself.
#[derive(Debug, Error)]
pub enum ChildMainError {
    #[error("SafeRuntime child protocol failed: {0}")]
    Protocol(String),
    #[error("SafeRuntime child expected Initialize as its first request")]
    ExpectedInitialization,
    #[error("SafeRuntime child received a second Initialize request")]
    AlreadyInitialized,
    #[error(
        "SafeRuntime child received execution {execution_id} before acknowledging its prior result"
    )]
    ExpectedAcknowledgement { execution_id: u64 },
    #[error("SafeRuntime child received acknowledgement for unknown execution {execution_id}")]
    UnknownExecution { execution_id: u64 },
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
#[derive(Debug)]
pub struct SafeRuntime {
    worker: Mutex<WorkerProcess>,
    transport: GpuTransport,
}

impl SafeRuntime {
    /// Construct a process-isolated runtime from Catena source paths.
    pub fn new<I>(paths: I, dialect: GpuDialect) -> Result<Self, SafeInitError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let sources = paths
            .into_iter()
            .map(|path| {
                fs::read_to_string(&path)
                    .map_err(|source| SafeInitError::ReadSource { path, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_owned_sources(sources, dialect)
    }

    /// Construct a process-isolated runtime from in-memory Catena sources.
    pub fn from_sources<'a, I>(sources: I, dialect: GpuDialect) -> Result<Self, SafeInitError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        Self::from_owned_sources(
            sources.into_iter().map(ToOwned::to_owned).collect(),
            dialect,
        )
    }

    fn from_owned_sources(
        sources: Vec<String>,
        dialect: GpuDialect,
    ) -> Result<Self, SafeInitError> {
        let transport = GpuTransport::new(dialect)?;
        let executable = env::current_exe().map_err(SafeInitError::CurrentExecutable)?;
        let mut worker = WorkerProcess::spawn(&executable)?;
        worker
            .send(&Request::Initialize {
                sources,
                dialect: WireGpuDialect::from(dialect),
            })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Initialized(Ok(())) => Ok(Self {
                worker: Mutex::new(worker),
                transport,
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            Response::Executed { .. } | Response::Acknowledged { .. } => {
                Err(SafeInitError::UnexpectedResponse)
            }
        }
    }

    /// Upload `u64` values into an ordinary parent-owned allocation.
    pub fn mem_u64(&self, values: &[u64]) -> Result<MemOwn, SafeMemoryError> {
        self.mem_from_bytes(slice_as_bytes(values))
    }

    /// Upload `f32` values into an ordinary parent-owned allocation.
    pub fn mem_f32(&self, values: &[f32]) -> Result<MemOwn, SafeMemoryError> {
        self.mem_from_bytes(slice_as_bytes(values))
    }

    fn mem_from_bytes(&self, bytes: &[u8]) -> Result<MemOwn, SafeMemoryError> {
        self.transport.allocate_from_host(bytes)?.into_mem_own()
    }

    /// Run a source-level program in the child process.
    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], SafeExecError> {
        let mut prepared = PreparedArguments::new(&self.transport, args.into())?;
        let wire_args = prepared.take_wire();
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| SafeExecError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(SafeExecError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }

        let execution_id = worker.next_execution_id();
        worker
            .send(&Request::Execute {
                execution_id,
                name: name.to_string(),
                args: wire_args,
            })
            .map_err(map_exec_worker_error)?;

        let results = match worker.receive().map_err(map_exec_worker_error)? {
            Response::Executed {
                execution_id: response_id,
                result,
            } if response_id == execution_id => match result {
                Ok(values) => values,
                Err(RemoteExecError::Runtime(error)) => {
                    return Err(SafeExecError::Runtime(error));
                }
                Err(RemoteExecError::Memory(error)) => {
                    return Err(SafeExecError::RemoteMemory(error));
                }
            },
            _ => return Err(SafeExecError::UnexpectedResponse),
        };

        let decoded = self.decode_results(results);
        let acknowledged = acknowledge_execution(&mut worker, execution_id);
        let values = match decoded {
            Ok(values) => {
                acknowledged?;
                values
            }
            Err(error) => {
                let _ = acknowledged;
                return Err(error);
            }
        };

        values
            .try_into()
            .map_err(|_| SafeExecError::UnexpectedResponse)
    }

    fn decode_results(
        &self,
        results: Vec<WireValue>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        results
            .into_iter()
            .map(|result| match result {
                WireValue::Bool(value) => Ok(Value::Bool(value)),
                WireValue::U32(value) => Ok(Value::U32(value)),
                WireValue::U64(value) => Ok(Value::U64(value)),
                WireValue::F32(value) => Ok(Value::F32(value)),
                WireValue::Mem(memory) => {
                    if memory.capability != WireCapability::Own {
                        return Err(SafeExecError::RemoteMemory(
                            "Runtime returned a cap.ref memory value".to_string(),
                        ));
                    }
                    let handle = decode_ipc_handle(memory)?;
                    let mapping = self.transport.open(handle)?;
                    let allocation = self.transport.copy_from_device(
                        mapping.as_ptr().cast_const(),
                        mapping.byte_len(),
                        mapping.dialect(),
                    )?;
                    drop(mapping);
                    Ok(Value::MemOwn(allocation.into_mem_own()?))
                }
            })
            .collect()
    }
}

fn acknowledge_execution(
    worker: &mut WorkerProcess,
    execution_id: u64,
) -> Result<(), SafeExecError> {
    worker
        .send(&Request::Acknowledge { execution_id })
        .map_err(map_exec_worker_error)?;
    match worker.receive().map_err(map_exec_worker_error)? {
        Response::Acknowledged {
            execution_id: response_id,
        } if response_id == execution_id => Ok(()),
        _ => Err(SafeExecError::UnexpectedResponse),
    }
}

struct PreparedArguments<'a> {
    wire: Vec<WireValue>,
    _staging: Vec<OwnedAllocation>,
    _values: Vec<Value<'a>>,
}

impl<'a> PreparedArguments<'a> {
    fn new(transport: &GpuTransport, values: Vec<Value<'a>>) -> Result<Self, SafeMemoryError> {
        let mut wire = Vec::with_capacity(values.len());
        let mut staging = Vec::new();
        for value in &values {
            let encoded = match value {
                Value::Bool(value) => WireValue::Bool(*value),
                Value::U32(value) => WireValue::U32(*value),
                Value::U64(value) => WireValue::U64(*value),
                Value::F32(value) => WireValue::F32(*value),
                Value::MemOwn(memory) => {
                    let allocation = transport.copy_from_device(
                        memory.as_ptr().cast_const(),
                        memory.byte_len(),
                        memory.dialect(),
                    )?;
                    let handle = allocation.export()?;
                    staging.push(allocation);
                    WireValue::Mem(encode_ipc_handle(WireCapability::Own, handle))
                }
                Value::MemRef(memory) => {
                    let allocation = transport.copy_from_device(
                        memory.as_ptr().cast_const(),
                        memory.byte_len(),
                        memory.dialect(),
                    )?;
                    let handle = allocation.export()?;
                    staging.push(allocation);
                    WireValue::Mem(encode_ipc_handle(WireCapability::Ref, handle))
                }
            };
            wire.push(encoded);
        }
        Ok(Self {
            wire,
            _staging: staging,
            _values: values,
        })
    }

    fn take_wire(&mut self) -> Vec<WireValue> {
        std::mem::take(&mut self.wire)
    }
}

fn encode_ipc_handle(capability: WireCapability, handle: IpcHandle) -> WireMemory {
    let dialect = handle.dialect();
    let byte_len = handle.byte_len();
    WireMemory {
        capability,
        dialect: dialect.into(),
        byte_len,
        handle: handle.into_bytes(),
    }
}

fn decode_ipc_handle(memory: WireMemory) -> Result<IpcHandle, SafeMemoryError> {
    IpcHandle::from_bytes(
        GpuDialect::from(memory.dialect),
        memory.byte_len,
        memory.handle,
    )
}

/// Run the SafeRuntime child loop when this executable was spawned as a worker.
///
/// Call this before argument parsing or writing to stdout. The return value is
/// `false` for a normal invocation and `true` after worker mode finishes.
pub fn run_safe_runtime_child_if_requested() -> Result<bool, ChildMainError> {
    if env::var_os(CHILD_MODE_ENV).is_none() {
        return Ok(false);
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    run_child_loop(stdin.lock(), stdout.lock())?;
    Ok(true)
}

fn run_child_loop(mut reader: impl Read, mut writer: impl io::Write) -> Result<(), ChildMainError> {
    let request = read_request(&mut reader)?.ok_or(ChildMainError::ExpectedInitialization)?;
    let Request::Initialize { sources, dialect } = request else {
        return Err(ChildMainError::ExpectedInitialization);
    };
    let dialect = GpuDialect::from(dialect);

    let source_refs = sources.iter().map(String::as_str);
    let runtime = match Runtime::from_sources(source_refs, dialect) {
        Ok(runtime) => runtime,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    let transport = match GpuTransport::new(dialect) {
        Ok(transport) => transport,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    write_response(&mut writer, &Response::Initialized(Ok(())))?;

    let mut pending_results: HashMap<u64, Vec<MemOwn>> = HashMap::new();
    while let Some(request) = read_request(&mut reader)? {
        match request {
            Request::Initialize { .. } => return Err(ChildMainError::AlreadyInitialized),
            Request::Shutdown => return Ok(()),
            Request::Acknowledge { execution_id } => {
                if pending_results.remove(&execution_id).is_none() {
                    return Err(ChildMainError::UnknownExecution { execution_id });
                }
                write_response(&mut writer, &Response::Acknowledged { execution_id })?;
            }
            Request::Execute {
                execution_id,
                name,
                args,
            } => {
                if !pending_results.is_empty() {
                    return Err(ChildMainError::ExpectedAcknowledgement { execution_id });
                }
                let result = execute_in_child(&runtime, &transport, &name, args);
                let response = match result {
                    Ok((values, owners)) => {
                        pending_results.insert(execution_id, owners);
                        Response::Executed {
                            execution_id,
                            result: Ok(values),
                        }
                    }
                    Err(error) => Response::Executed {
                        execution_id,
                        result: Err(error),
                    },
                };
                write_response(&mut writer, &response)?;
            }
        }
    }
    Ok(())
}

fn execute_in_child(
    runtime: &Runtime,
    transport: &GpuTransport,
    name: &str,
    wire_args: Vec<WireValue>,
) -> Result<(Vec<WireValue>, Vec<MemOwn>), RemoteExecError> {
    let imported = import_arguments(transport, wire_args)
        .map_err(|error| RemoteExecError::Memory(error.to_string()))?;
    let ImportedArguments {
        plans,
        mut owned,
        borrowed,
    } = imported;
    let mut args = Vec::with_capacity(plans.len());
    for plan in plans {
        let value = match plan {
            ChildArgument::Bool(value) => Value::Bool(value),
            ChildArgument::U32(value) => Value::U32(value),
            ChildArgument::U64(value) => Value::U64(value),
            ChildArgument::F32(value) => Value::F32(value),
            ChildArgument::Own(index) => Value::MemOwn(
                owned[index]
                    .take()
                    .expect("owned argument plan should be used exactly once"),
            ),
            ChildArgument::Ref(index) => {
                let mapping = &borrowed[index];
                // SAFETY: `mapping` remains alive until after the synchronous
                // Runtime call below and guards this exact imported region.
                Value::MemRef(unsafe {
                    MemRef::from_raw_parts(
                        mapping.as_ptr(),
                        mapping.byte_len(),
                        mapping.dialect(),
                        mapping,
                    )
                })
            }
        };
        args.push(value);
    }

    let values = runtime
        .exec_values(name, args)
        .map_err(RemoteExecError::Runtime)?;
    transport
        .synchronize()
        .map_err(|error| RemoteExecError::Memory(error.to_string()))?;
    export_results(transport, values).map_err(|error| RemoteExecError::Memory(error.to_string()))
}

struct ImportedArguments {
    plans: Vec<ChildArgument>,
    owned: Vec<Option<MemOwn>>,
    borrowed: Vec<IpcMapping>,
}

enum ChildArgument {
    Bool(u8),
    U32(u32),
    U64(u64),
    F32(f32),
    Own(usize),
    Ref(usize),
}

fn import_arguments(
    transport: &GpuTransport,
    wire_args: Vec<WireValue>,
) -> Result<ImportedArguments, SafeMemoryError> {
    let mut plans = Vec::with_capacity(wire_args.len());
    let mut owned = Vec::new();
    let mut borrowed = Vec::new();
    for wire in wire_args {
        let plan = match wire {
            WireValue::Bool(value) => ChildArgument::Bool(value),
            WireValue::U32(value) => ChildArgument::U32(value),
            WireValue::U64(value) => ChildArgument::U64(value),
            WireValue::F32(value) => ChildArgument::F32(value),
            WireValue::Mem(memory) => {
                let capability = memory.capability;
                let mapping = transport.open(decode_ipc_handle(memory)?)?;
                match capability {
                    WireCapability::Own => {
                        let allocation = transport.copy_from_device(
                            mapping.as_ptr().cast_const(),
                            mapping.byte_len(),
                            mapping.dialect(),
                        )?;
                        drop(mapping);
                        let index = owned.len();
                        owned.push(Some(allocation.into_mem_own()?));
                        ChildArgument::Own(index)
                    }
                    WireCapability::Ref => {
                        let index = borrowed.len();
                        borrowed.push(mapping);
                        ChildArgument::Ref(index)
                    }
                }
            }
        };
        plans.push(plan);
    }
    Ok(ImportedArguments {
        plans,
        owned,
        borrowed,
    })
}

fn export_results(
    transport: &GpuTransport,
    values: Vec<Value<'static>>,
) -> Result<(Vec<WireValue>, Vec<MemOwn>), SafeMemoryError> {
    let mut wire = Vec::with_capacity(values.len());
    let mut owners = Vec::new();
    for value in values {
        let result = match value {
            Value::Bool(value) => WireValue::Bool(value),
            Value::U32(value) => WireValue::U32(value),
            Value::U64(value) => WireValue::U64(value),
            Value::F32(value) => WireValue::F32(value),
            Value::MemOwn(memory) => {
                let handle =
                    transport.export(memory.as_ptr(), memory.byte_len(), memory.dialect())?;
                owners.push(memory);
                WireValue::Mem(encode_ipc_handle(WireCapability::Own, handle))
            }
            Value::MemRef(_) => return Err(SafeMemoryError::UnsupportedRefOutput),
        };
        wire.push(result);
    }
    Ok((wire, owners))
}

fn read_request(reader: &mut impl Read) -> Result<Option<Request>, ChildMainError> {
    read_frame(reader).map_err(child_protocol_error)
}

fn write_response(writer: &mut impl io::Write, response: &Response) -> Result<(), ChildMainError> {
    write_frame(writer, response).map_err(child_protocol_error)
}

fn child_protocol_error(error: ProtocolError) -> ChildMainError {
    ChildMainError::Protocol(error.to_string())
}

#[derive(Debug, Clone)]
struct Termination {
    status: ExitStatus,
    stderr: String,
}

#[derive(Debug)]
struct WorkerProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_reader: Option<JoinHandle<Vec<u8>>>,
    termination: Option<Termination>,
    next_execution_id: u64,
}

#[derive(Debug, Error)]
enum WorkerError {
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("failed to wait for SafeRuntime child: {0}")]
    Wait(#[source] io::Error),
    #[error("SafeRuntime child terminated")]
    Terminated(Termination),
}

impl WorkerProcess {
    fn spawn(executable: &Path) -> Result<Self, SafeInitError> {
        let mut child = Command::new(executable)
            .env(CHILD_MODE_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| SafeInitError::Spawn {
                executable: executable.to_path_buf(),
                source,
            })?;
        let stdin = child
            .stdin
            .take()
            .expect("piped child stdin should be available");
        let stdout = child
            .stdout
            .take()
            .expect("piped child stdout should be available");
        let mut stderr = child
            .stderr
            .take()
            .expect("piped child stderr should be available");
        let stderr_reader = thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stderr.read_to_end(&mut bytes);
            bytes
        });

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            stderr_reader: Some(stderr_reader),
            termination: None,
            next_execution_id: 1,
        })
    }

    fn next_execution_id(&mut self) -> u64 {
        let id = self.next_execution_id;
        self.next_execution_id = self.next_execution_id.wrapping_add(1);
        id
    }

    fn send(&mut self, request: &Request) -> Result<(), WorkerError> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(WorkerError::Terminated(
                self.termination
                    .clone()
                    .expect("closed worker stdin should have termination state"),
            ));
        };
        write_frame(stdin, request).map_err(WorkerError::Protocol)
    }

    fn receive(&mut self) -> Result<Response, WorkerError> {
        match read_frame(&mut self.stdout).map_err(WorkerError::Protocol)? {
            Some(response) => Ok(response),
            None => Err(WorkerError::Terminated(self.reap()?)),
        }
    }

    fn termination(&self) -> Option<&Termination> {
        self.termination.as_ref()
    }

    fn reap(&mut self) -> Result<Termination, WorkerError> {
        if let Some(termination) = &self.termination {
            return Ok(termination.clone());
        }
        self.stdin.take();
        let status = self.child.wait().map_err(WorkerError::Wait)?;
        let stderr = self.take_stderr();
        let termination = Termination { status, stderr };
        self.termination = Some(termination.clone());
        Ok(termination)
    }

    fn take_stderr(&mut self) -> String {
        let bytes = self
            .stderr_reader
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.termination.is_none() {
            if let Some(stdin) = self.stdin.as_mut() {
                let _ = write_frame(stdin, &Request::Shutdown);
            }
            self.stdin.take();
            let _ = self.child.wait();
        }
        if self.stderr_reader.is_some() {
            let _ = self.take_stderr();
        }
    }
}

fn map_init_worker_error(error: WorkerError) -> SafeInitError {
    match error {
        WorkerError::Terminated(termination) => SafeInitError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeInitError::Transport(other.to_string()),
    }
}

fn map_exec_worker_error(error: WorkerError) -> SafeExecError {
    match error {
        WorkerError::Terminated(termination) => SafeExecError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeExecError::Transport(other.to_string()),
    }
}

fn slice_as_bytes<T>(values: &[T]) -> &[u8] {
    let byte_len = std::mem::size_of_val(values);
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }
}
