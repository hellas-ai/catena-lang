//! Process-isolated execution for Catena programs.
//!
//! Generated GPU code runs through native libraries, so a failed Catena assertion,
//! GPU runtime failure, or native crash can terminate the process rather than return
//! a Rust error. [`SafeRuntime`] solves this by executing [`Runtime`] in a child
//! process and communicating over a framed protocol. If execution terminates the
//! child, the host process survives and receives a structured error with its exit
//! status and stderr.

use std::{
    env, fs,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};

use thiserror::Error;

mod ipc;
mod protocol;

use self::{
    ipc::{ImportedIpcAllocation, IpcMemoryHandle, IpcTransport},
    protocol::{
        ProtocolError, RemoteExecError, Request, Response, WireExecution, WireIpcBuffer, WireValue,
        read_frame, write_frame,
    },
};
use crate::{
    codegen::GpuDialect,
    runtime::{ExecError, MemError, MemOwn, Runtime, Value},
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
    Memory(#[from] MemError),
}

/// Execution failures reported by [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeExecError {
    #[error(transparent)]
    Runtime(#[from] ExecError),
    #[error("SafeRuntime transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child returned an unexpected execution response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] MemError),
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
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
#[derive(Debug)]
pub struct SafeRuntime {
    worker: Mutex<WorkerProcess>,
    ipc: IpcTransport,
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
        let executable = env::current_exe().map_err(SafeInitError::CurrentExecutable)?;
        let ipc = IpcTransport::load(dialect)?;
        let mut worker = WorkerProcess::spawn(&executable)?;
        worker
            .send(&Request::Initialize { sources, dialect })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Initialized(Ok(())) => Ok(Self {
                worker: Mutex::new(worker),
                ipc,
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            Response::Executed(_) => Err(SafeInitError::UnexpectedResponse),
        }
    }

    /// Run a source-level program in the child process.
    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], SafeExecError> {
        self.exec_values(name, args.into())?
            .try_into()
            .map_err(|_| SafeExecError::UnexpectedResponse)
    }

    /// Run a source-level program with dynamically sized inputs and outputs.
    pub fn exec_values<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        let (buffers, wire_args) = self.encode_parent_arguments(&args)?;
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

        worker
            .send(&Request::Execute {
                name: name.to_string(),
                buffers,
                args: wire_args,
            })
            .map_err(map_exec_worker_error)?;

        let response = worker.receive().map_err(map_exec_worker_error)?;
        let execution = match response {
            Response::Executed(Ok(execution)) => execution,
            Response::Executed(Err(RemoteExecError::Runtime(error))) => {
                return Err(SafeExecError::Runtime(error));
            }
            Response::Executed(Err(RemoteExecError::Memory(error))) => {
                return Err(SafeExecError::Transport(format!(
                    "child memory IPC failed: {error}"
                )));
            }
            Response::Initialized(_) => {
                return Err(SafeExecError::UnexpectedResponse);
            }
        };
        let values = self.decode_child_outputs(execution);
        worker
            .send(&Request::ReleaseOutputs)
            .map_err(map_exec_worker_error)?;
        values
    }

    /// Exports parent-owned arguments as views for the child to copy into its own allocations.
    fn encode_parent_arguments(
        &self,
        args: &[Value<'_>],
    ) -> Result<(Vec<WireIpcBuffer>, Vec<WireValue>), SafeExecError> {
        let mut buffers = Vec::new();
        let mut values = Vec::with_capacity(args.len());
        for (index, value) in args.iter().enumerate() {
            let wire = match value {
                Value::Bool(value) => WireValue::Bool(*value),
                Value::U16(value) => WireValue::U16(*value),
                Value::U32(value) => WireValue::U32(*value),
                Value::U64(value) => WireValue::U64(*value),
                Value::F32(value) => WireValue::F32(*value),
                Value::MemOwn(memory) => {
                    if memory.dialect() != self.ipc.dialect() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = self.ipc.export_view(memory.as_ref())?;
                    let buffer_index = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                    WireValue::MemOwn {
                        buffer: buffer_index,
                        view_offset: exported.view_offset(),
                        byte_len: exported.byte_len(),
                    }
                }
                Value::MemRef(memory) => {
                    if memory.dialect() != self.ipc.dialect() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = self.ipc.export_view(*memory)?;
                    let buffer_index = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                    WireValue::MemRef {
                        buffer: buffer_index,
                        view_offset: exported.view_offset(),
                        byte_len: exported.byte_len(),
                    }
                }
            };
            values.push(wire);
        }
        if !buffers.is_empty() {
            self.ipc.synchronize()?;
        }
        Ok((buffers, values))
    }

    /// Copies child-owned outputs into parent-owned allocations before releasing them remotely.
    fn decode_child_outputs(
        &self,
        execution: WireExecution,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        let imported = import_ipc_buffers(&self.ipc, execution.buffers).map_err(|error| {
            SafeExecError::Transport(format!("child memory IPC failed: {error}"))
        })?;
        execution
            .values
            .into_iter()
            .map(|value| match value {
                WireValue::Bool(value) => Ok(Value::Bool(value)),
                WireValue::U16(value) => Ok(Value::U16(value)),
                WireValue::U32(value) => Ok(Value::U32(value)),
                WireValue::U64(value) => Ok(Value::U64(value)),
                WireValue::F32(value) => Ok(Value::F32(value)),
                WireValue::MemOwn {
                    buffer,
                    view_offset,
                    byte_len,
                } => imported
                    .get(buffer)
                    .ok_or_else(|| SafeExecError::Transport("invalid IPC memory view".to_string()))?
                    .copy_view_into_owned(view_offset, byte_len)?
                    .map(Value::MemOwn)
                    .ok_or_else(|| SafeExecError::Transport("invalid IPC memory view".to_string())),
                WireValue::MemRef { .. } => Err(SafeExecError::UnexpectedResponse),
            })
            .collect()
    }
}

fn encode_ipc_buffer(exported: ipc::ExportedIpcView) -> WireIpcBuffer {
    WireIpcBuffer {
        handle: exported.handle().map(|handle| handle.as_bytes().to_vec()),
        allocation_byte_len: exported.allocation_byte_len(),
    }
}

fn intern_buffer(buffers: &mut Vec<WireIpcBuffer>, buffer: WireIpcBuffer) -> usize {
    buffers
        .iter()
        .position(|existing| existing == &buffer)
        .unwrap_or_else(|| {
            buffers.push(buffer);
            buffers.len() - 1
        })
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

    let source_refs = sources.iter().map(String::as_str);
    let runtime = match Runtime::from_sources(source_refs, dialect) {
        Ok(runtime) => runtime,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    let ipc = IpcTransport::from_runtime(&runtime);
    write_response(&mut writer, &Response::Initialized(Ok(())))?;

    let mut pending_outputs = Vec::new();
    while let Some(request) = read_request(&mut reader)? {
        match request {
            Request::Initialize { .. } => return Err(ChildMainError::AlreadyInitialized),
            Request::Shutdown => return Ok(()),
            Request::ReleaseOutputs => {
                pending_outputs.clear();
            }
            Request::Execute {
                name,
                buffers,
                args,
            } => {
                let response = if pending_outputs.is_empty() {
                    execute_in_child(&runtime, &ipc, &name, buffers, args, &mut pending_outputs)
                } else {
                    Response::Executed(Err(RemoteExecError::Memory(
                        "previous owned outputs have not been released".to_string(),
                    )))
                };
                write_response(&mut writer, &response)?;
            }
        }
    }

    Ok(())
}

/// Copies owned arguments into the child, runs the program, and prepares its outputs for export.
fn execute_in_child(
    runtime: &Runtime,
    ipc: &IpcTransport,
    name: &str,
    buffers: Vec<WireIpcBuffer>,
    wire_args: Vec<WireValue>,
    pending_outputs: &mut Vec<MemOwn>,
) -> Response {
    let imported = match import_ipc_buffers(ipc, buffers) {
        Ok(imported) => imported,
        Err(error) => return Response::Executed(Err(RemoteExecError::Memory(error))),
    };

    let args = match wire_args
        .into_iter()
        .map(|value| match value {
            WireValue::Bool(value) => Ok(Value::Bool(value)),
            WireValue::U16(value) => Ok(Value::U16(value)),
            WireValue::U32(value) => Ok(Value::U32(value)),
            WireValue::U64(value) => Ok(Value::U64(value)),
            WireValue::F32(value) => Ok(Value::F32(value)),
            WireValue::MemRef {
                buffer,
                view_offset,
                byte_len,
            } => imported
                .get(buffer)
                .and_then(|allocation| allocation.as_mem_ref(view_offset, byte_len))
                .map(Value::MemRef)
                .ok_or_else(|| "invalid IPC memory view".to_string()),
            WireValue::MemOwn {
                buffer,
                view_offset,
                byte_len,
            } => imported
                .get(buffer)
                .ok_or_else(|| "invalid IPC memory view".to_string())?
                .copy_view_into_owned(view_offset, byte_len)
                .map_err(|error| error.to_string())?
                .map(Value::MemOwn)
                .ok_or_else(|| "invalid IPC memory view".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(args) => args,
        Err(error) => return Response::Executed(Err(RemoteExecError::Memory(error))),
    };

    let values = match runtime.exec_values(name, args) {
        Ok(values) => values,
        Err(error) => return Response::Executed(Err(RemoteExecError::Runtime(error))),
    };
    if let Err(error) = ipc.synchronize() {
        return Response::Executed(Err(RemoteExecError::Memory(error.to_string())));
    }
    match encode_child_outputs(ipc, values, pending_outputs) {
        Ok(execution) => Response::Executed(Ok(execution)),
        Err(error) => {
            pending_outputs.clear();
            Response::Executed(Err(error))
        }
    }
}

fn import_ipc_buffers(
    ipc: &IpcTransport,
    buffers: Vec<WireIpcBuffer>,
) -> Result<Vec<ImportedIpcAllocation>, String> {
    buffers
        .into_iter()
        .map(|buffer| {
            if buffer.handle.is_none() && buffer.allocation_byte_len != 0 {
                return Err("non-empty IPC allocation has no handle".to_string());
            }
            let handle =
                buffer
                    .handle
                    .map(|bytes| {
                        bytes.try_into().map(IpcMemoryHandle::from_bytes).map_err(
                            |bytes: Vec<u8>| format!("IPC memory handle has {} bytes", bytes.len()),
                        )
                    })
                    .transpose()?;
            ipc.import_allocation(handle, buffer.allocation_byte_len)
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// Exports owned outputs and retains them until the parent confirms it has copied them.
fn encode_child_outputs(
    ipc: &IpcTransport,
    values: Vec<Value<'static>>,
    pending_outputs: &mut Vec<MemOwn>,
) -> Result<WireExecution, RemoteExecError> {
    let mut buffers = Vec::new();
    let mut wire_values = Vec::with_capacity(values.len());
    for value in values {
        let wire = match value {
            Value::Bool(value) => WireValue::Bool(value),
            Value::U16(value) => WireValue::U16(value),
            Value::U32(value) => WireValue::U32(value),
            Value::U64(value) => WireValue::U64(value),
            Value::F32(value) => WireValue::F32(value),
            Value::MemOwn(memory) => {
                let exported = ipc
                    .export_view(memory.as_ref())
                    .map_err(|error| RemoteExecError::Memory(error.to_string()))?;
                let buffer = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                let value = WireValue::MemOwn {
                    buffer,
                    view_offset: exported.view_offset(),
                    byte_len: exported.byte_len(),
                };
                pending_outputs.push(memory);
                value
            }
            Value::MemRef(_) => unreachable!("Runtime rejects borrowed memory outputs"),
        };
        wire_values.push(wire);
    }
    Ok(WireExecution {
        buffers,
        values: wire_values,
    })
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
        })
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
