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

mod protocol;

use self::protocol::{
    ProtocolError, RemoteExecError, Request, Response, WireArgument, WireGpuDialect, WireResult,
    read_frame, write_frame,
};
use crate::{
    codegen::GpuDialect,
    runtime::{DeviceAllocator, DeviceBuffer, ExecError, MemError, Runtime, Value},
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
    #[error(transparent)]
    Memory(#[from] MemError),
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
    #[error("SafeRuntime child expected pending result transfers to be released")]
    ExpectedResultTransferRelease,
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
#[derive(Debug)]
pub struct SafeRuntime {
    worker: Mutex<WorkerProcess>,
    device_allocator: DeviceAllocator,
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
        let mut worker = WorkerProcess::spawn(&executable)?;
        let device_allocator = DeviceAllocator::new(dialect)?;
        worker
            .send(&Request::Initialize {
                sources,
                dialect: WireGpuDialect::from(dialect),
            })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Initialized(Ok(())) => Ok(Self {
                worker: Mutex::new(worker),
                device_allocator,
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            Response::Executed(_) | Response::ResultTransfersReleased => {
                Err(SafeInitError::UnexpectedResponse)
            }
        }
    }

    /// Upload `u64` values to parent-owned device memory.
    pub fn mem_u64(&self, values: &[u64]) -> Result<Value, MemError> {
        self.device_allocator
            .allocate_from_bytes(slice_as_bytes(values))
            .map(Value::from)
    }

    /// Upload `f32` values to parent-owned device memory.
    pub fn mem_f32(&self, values: &[f32]) -> Result<Value, MemError> {
        self.device_allocator
            .allocate_from_bytes(slice_as_bytes(values))
            .map(Value::from)
    }

    /// Return the parent-side allocator whose buffers can be passed to this runtime.
    pub fn device_allocator(&self) -> &DeviceAllocator {
        &self.device_allocator
    }

    /// Run a source-level program in the child process.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], SafeExecError> {
        let wire_args = args
            .iter()
            .map(WireArgument::from_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(SafeExecError::Memory)?;
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
                args: wire_args,
                output_count: N,
            })
            .map_err(map_exec_worker_error)?;

        let response = worker.receive().map_err(map_exec_worker_error)?;
        let values = match response {
            Response::Executed(Ok(values)) => self.decode_results(values, &args, &mut worker)?,
            Response::Executed(Err(RemoteExecError::Runtime(error))) => {
                return Err(SafeExecError::Runtime(error));
            }
            Response::Executed(Err(RemoteExecError::Memory(error))) => {
                return Err(SafeExecError::RemoteMemory(error));
            }
            Response::Initialized(_) | Response::ResultTransfersReleased => {
                return Err(SafeExecError::UnexpectedResponse);
            }
        };
        values
            .try_into()
            .map_err(|_| SafeExecError::UnexpectedResponse)
    }

    fn decode_results(
        &self,
        results: Vec<WireResult>,
        args: &[Value],
        worker: &mut WorkerProcess,
    ) -> Result<Vec<Value>, SafeExecError> {
        let has_result_transfers = results
            .iter()
            .any(|result| matches!(result, WireResult::Mem(_)));
        let mut values = Vec::with_capacity(results.len());
        let decoded = (|| {
            for result in results {
                let value = match result {
                    WireResult::Bool(value) => Value::Bool(value),
                    WireResult::U32(value) => Value::U32(value),
                    WireResult::U64(value) => Value::U64(value),
                    WireResult::F32(value) => Value::F32(value),
                    WireResult::Mem(memory) => {
                        let (source, view_offset, byte_len) =
                            memory.import(&self.device_allocator)?;
                        let destination = self.device_allocator.allocate(source.byte_len())?;
                        destination.copy_from_device(0, &source, 0, source.byte_len())?;
                        Value::Mem(destination.into_mem_view(view_offset, byte_len)?)
                    }
                    WireResult::ArgumentAlias {
                        argument_index,
                        view_offset,
                        byte_len,
                    } => alias_view(args.get(argument_index), view_offset, byte_len)?,
                    WireResult::ResultAlias {
                        result_index,
                        view_offset,
                        byte_len,
                    } => alias_view(values.get(result_index), view_offset, byte_len)?,
                };
                values.push(value);
            }
            Ok::<_, MemError>(())
        })();

        if has_result_transfers {
            worker
                .send(&Request::ReleaseResultTransfers)
                .map_err(map_exec_worker_error)?;
            match worker.receive().map_err(map_exec_worker_error)? {
                Response::ResultTransfersReleased => {}
                _ => return Err(SafeExecError::UnexpectedResponse),
            }
        }

        decoded.map(|()| values).map_err(SafeExecError::Memory)
    }
}

fn alias_view(value: Option<&Value>, offset: u64, byte_len: u64) -> Result<Value, MemError> {
    let Some(Value::Mem(memory)) = value else {
        return Err(MemError::InvalidRemoteMemory(
            "alias does not refer to an available memory value".to_string(),
        ));
    };
    memory.view(offset, byte_len).map(Value::Mem)
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
    let runtime = match Runtime::from_sources(source_refs, dialect.into()) {
        Ok(runtime) => {
            write_response(&mut writer, &Response::Initialized(Ok(())))?;
            runtime
        }
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };

    let mut pending_result_transfers = Vec::new();
    while let Some(request) = read_request(&mut reader)? {
        match request {
            Request::Initialize { .. } => return Err(ChildMainError::AlreadyInitialized),
            Request::Shutdown => return Ok(()),
            Request::ReleaseResultTransfers => {
                pending_result_transfers.clear();
                write_response(&mut writer, &Response::ResultTransfersReleased)?;
            }
            Request::Execute {
                name,
                args,
                output_count,
            } => {
                if !pending_result_transfers.is_empty() {
                    return Err(ChildMainError::ExpectedResultTransferRelease);
                }
                let response = execute_in_child(
                    &runtime,
                    &name,
                    args,
                    output_count,
                    &mut pending_result_transfers,
                );
                write_response(&mut writer, &response)?;
            }
        }
    }

    Ok(())
}

fn execute_in_child(
    runtime: &Runtime,
    name: &str,
    wire_args: Vec<WireArgument>,
    output_count: usize,
    pending_result_transfers: &mut Vec<DeviceBuffer>,
) -> Response {
    let args = match import_arguments(runtime.device_allocator(), wire_args) {
        Ok(args) => args,
        Err(error) => {
            return Response::Executed(Err(RemoteExecError::Memory(error.to_string())));
        }
    };

    let result_inputs = args.clone();
    match runtime.exec_values(name, args, output_count) {
        Ok(values) => {
            if let Err(error) = runtime.device_allocator().synchronize() {
                return Response::Executed(Err(RemoteExecError::Memory(error.to_string())));
            }
            match export_results(&result_inputs, &values, pending_result_transfers) {
                Ok(values) => Response::Executed(Ok(values)),
                Err(error) => Response::Executed(Err(RemoteExecError::Memory(error.to_string()))),
            }
        }
        Err(error) => Response::Executed(Err(RemoteExecError::Runtime(error))),
    }
}

fn import_arguments(
    allocator: &DeviceAllocator,
    arguments: Vec<WireArgument>,
) -> Result<Vec<Value>, MemError> {
    arguments
        .into_iter()
        .map(|argument| match argument {
            WireArgument::Bool(value) => Ok(Value::Bool(value)),
            WireArgument::U32(value) => Ok(Value::U32(value)),
            WireArgument::U64(value) => Ok(Value::U64(value)),
            WireArgument::F32(value) => Ok(Value::F32(value)),
            WireArgument::Mem(memory) => {
                let (buffer, offset, byte_len) = memory.import(allocator)?;
                buffer.into_mem_view(offset, byte_len).map(Value::Mem)
            }
        })
        .collect()
}

fn export_results(
    arguments: &[Value],
    values: &[Value],
    pending_result_transfers: &mut Vec<DeviceBuffer>,
) -> Result<Vec<WireResult>, MemError> {
    let mut results = Vec::with_capacity(values.len());
    let converted = (|| {
        for (result_index, value) in values.iter().enumerate() {
            let result = match value {
                Value::Bool(value) => WireResult::Bool(*value),
                Value::U32(value) => WireResult::U32(*value),
                Value::U64(value) => WireResult::U64(*value),
                Value::F32(value) => WireResult::F32(*value),
                Value::Mem(memory) => {
                    let buffer = memory.device_buffer();
                    let view_offset = buffer.view_offset(memory.abi.data, memory.abi.len)?;
                    if let Some(argument_index) = arguments
                        .iter()
                        .position(|argument| same_memory_allocation(argument, buffer))
                    {
                        WireResult::ArgumentAlias {
                            argument_index,
                            view_offset,
                            byte_len: memory.byte_len(),
                        }
                    } else if let Some(previous) = values[..result_index]
                        .iter()
                        .position(|result| same_memory_allocation(result, buffer))
                    {
                        WireResult::ResultAlias {
                            result_index: previous,
                            view_offset,
                            byte_len: memory.byte_len(),
                        }
                    } else {
                        let wire_memory = self::protocol::WireMemory::from_mem(memory)?;
                        pending_result_transfers.push(buffer.clone());
                        WireResult::Mem(wire_memory)
                    }
                }
            };
            results.push(result);
        }
        Ok::<_, MemError>(())
    })();
    if let Err(error) = converted {
        pending_result_transfers.clear();
        return Err(error);
    }
    Ok(results)
}

fn same_memory_allocation(value: &Value, buffer: &DeviceBuffer) -> bool {
    matches!(value, Value::Mem(memory) if memory.device_buffer().same_allocation(buffer))
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

fn slice_as_bytes<T>(values: &[T]) -> &[u8] {
    let byte_len = std::mem::size_of_val(values);
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }
}
