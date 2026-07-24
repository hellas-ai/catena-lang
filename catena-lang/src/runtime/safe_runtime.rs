use std::{
    env, fs,
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};

use thiserror::Error;

use super::{
    protocol::{
        ProtocolError, RemoteExecError, Request, Response, WireGpuDialect, WireValue, read_frame,
        write_frame,
    },
    runtime::{ExecError, Runtime},
    value::{Value, ValueKind},
};
use crate::codegen::GpuDialect;

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
}

/// Execution failures reported by [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeExecError {
    #[error(transparent)]
    Runtime(#[from] ExecError),
    #[error("SafeRuntime does not yet support {0:?} values")]
    UnsupportedValueKind(ValueKind),
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
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
#[derive(Debug)]
pub struct SafeRuntime {
    worker: Mutex<WorkerProcess>,
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
        worker
            .send(&Request::Initialize {
                sources,
                dialect: WireGpuDialect::from(dialect),
            })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Initialized(Ok(())) => Ok(Self {
                worker: Mutex::new(worker),
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            Response::Executed(_) => Err(SafeInitError::UnexpectedResponse),
        }
    }

    /// Run a source-level program in the child process.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], SafeExecError> {
        let args = args
            .into_iter()
            .map(WireValue::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(SafeExecError::UnsupportedValueKind)?;
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
                args,
                output_count: N,
            })
            .map_err(map_exec_worker_error)?;

        let response = worker.receive().map_err(map_exec_worker_error)?;
        let values = match response {
            Response::Executed(Ok(values)) => {
                values.into_iter().map(Value::from).collect::<Vec<_>>()
            }
            Response::Executed(Err(RemoteExecError::Runtime(error))) => {
                return Err(SafeExecError::Runtime(error));
            }
            Response::Executed(Err(RemoteExecError::UnsupportedValueKind(kind))) => {
                return Err(SafeExecError::UnsupportedValueKind(kind));
            }
            Response::Initialized(_) => return Err(SafeExecError::UnexpectedResponse),
        };
        values
            .try_into()
            .map_err(|_| SafeExecError::UnexpectedResponse)
    }
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

    while let Some(request) = read_request(&mut reader)? {
        match request {
            Request::Initialize { .. } => return Err(ChildMainError::AlreadyInitialized),
            Request::Shutdown => return Ok(()),
            Request::Execute {
                name,
                args,
                output_count,
            } => {
                let args = args.into_iter().map(Value::from).collect::<Vec<_>>();
                let response = match runtime.exec_values(&name, args, output_count) {
                    Ok(values) => {
                        match values
                            .into_iter()
                            .map(WireValue::try_from)
                            .collect::<Result<Vec<_>, _>>()
                        {
                            Ok(values) => Response::Executed(Ok(values)),
                            Err(kind) => {
                                Response::Executed(Err(RemoteExecError::UnsupportedValueKind(kind)))
                            }
                        }
                    }
                    Err(error) => Response::Executed(Err(RemoteExecError::Runtime(error))),
                };
                write_response(&mut writer, &response)?;
            }
        }
    }

    Ok(())
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
