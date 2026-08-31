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
    fs::File,
    io::{self, BufReader, Read},
    net::Shutdown,
    os::fd::{AsRawFd, FromRawFd, RawFd},
    os::unix::{net::UnixStream, process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};

use thiserror::Error;

mod assets;
mod fd_transport;
mod ipc;
mod protocol;
pub(crate) mod resident;

use self::{
    assets::{AssetStore, MAX_ASSET_BYTES, validate_byte_len},
    fd_transport::{FdTransportError, receive_file, send_file},
    ipc::{ImportedIpcAllocation, IpcMemoryHandle, IpcTransport},
    protocol::{
        ProtocolError, RemoteExecError, Request, ResidentResponse, Response, WireAssetSlice,
        WireExecution, WireIpcBuffer, WireModelBinding, WireValue, read_frame, write_frame,
    },
    resident::ResidentStore,
};
use crate::{
    codegen::GpuDialect,
    runtime::{Artifact, ExecError, MemError, MemOwn, Runtime, RuntimeId, Value},
};

const CHILD_MODE_ENV: &str = "CATENA_SAFE_RUNTIME_CHILD";
const CHILD_ASSET_SOCKET_FD: RawFd = 3;

/// Failures while attaching or slicing a session-resident asset.
#[derive(Debug, Error)]
pub enum AssetError {
    #[error("failed to inspect verified asset descriptor: {0}")]
    FileMetadata(#[source] io::Error),
    #[error("asset length must be between 1 and {maximum} bytes, got {actual}")]
    InvalidLength { actual: u64, maximum: u64 },
    #[error("asset transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child rejected asset: {0}")]
    Remote(String),
    #[error("SafeRuntime child returned an unexpected asset response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated while attaching an asset with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error("asset belongs to a different GPU session")]
    WrongSession,
    #[error(
        "asset slice at offset {offset} with {byte_len} bytes exceeds its {asset_byte_len}-byte asset"
    )]
    InvalidSlice {
        offset: u64,
        byte_len: u64,
        asset_byte_len: u64,
    },
}

impl AssetError {
    /// Whether the worker protocol can no longer be trusted after this error.
    ///
    /// Caller-side validation and a structured rejection from the worker leave
    /// the session usable. Transport failures, termination, and an unexpected
    /// response do not: the next operation must use a new session.
    pub fn invalidates_session(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::UnexpectedResponse
                | Self::ChildTerminated { .. }
                | Self::Unavailable { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttachedAsset {
    pub(crate) id: u64,
    pub(crate) byte_len: u64,
}

/// Failures from the worker-resident model and generation protocol.
#[derive(Debug, Error)]
pub enum ResidentError {
    #[error("resident GPU request failed: {0}")]
    Remote(String),
    #[error("SafeRuntime child returned an unexpected resident response")]
    UnexpectedResponse,
    #[error("resident GPU transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child terminated during a resident GPU request with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
}

impl ResidentError {
    /// Whether the worker protocol can no longer be trusted after this error.
    pub fn invalidates_session(&self) -> bool {
        !matches!(self, Self::Remote(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentModel {
    pub(crate) id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentGeneration {
    pub(crate) id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentAssetSlice {
    pub(crate) asset: u64,
    pub(crate) offset: u64,
    pub(crate) byte_len: u64,
}

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
    #[error("failed to create SafeRuntime asset transport: {0}")]
    AssetTransport(#[source] io::Error),
    #[error("SafeRuntime setup transport failed: {0}")]
    Transport(String),
    #[error("SafeRuntime child initialization failed: {0}")]
    RemoteInitialization(String),
    #[error("SafeRuntime child failed to load sources: {0}")]
    RemoteLoad(String),
    #[error("SafeRuntime child returned an unexpected setup response")]
    UnexpectedResponse,
    #[error(
        "SafeRuntime child terminated during initialization or loading with {status}: {stderr}"
    )]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] MemError),
}

impl SafeInitError {
    /// Whether an existing worker protocol can no longer be trusted after this
    /// error.
    ///
    /// Of these variants, only source reading and a structured load rejection
    /// can occur without invalidating a live session. The construction-only
    /// variants are conservatively classified as invalidating so callers never
    /// retain a session whose health is uncertain.
    pub fn invalidates_session(&self) -> bool {
        !matches!(self, Self::ReadSource { .. } | Self::RemoteLoad(_))
    }
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
    #[error("SafeRuntime child GPU synchronization failed: {0}")]
    GpuSynchronization(String),
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
    runtime_id: RuntimeId,
}

impl SafeRuntime {
    /// Construct an empty process-isolated runtime.
    pub fn new(dialect: GpuDialect) -> Result<Self, SafeInitError> {
        let executable = env::current_exe().map_err(SafeInitError::CurrentExecutable)?;
        let ipc = IpcTransport::load(dialect)?;
        let mut worker = WorkerProcess::spawn(&executable)?;
        worker
            .send(&Request::Initialize { dialect })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Initialized(Ok(())) => Ok(Self {
                worker: Mutex::new(worker),
                ipc,
                runtime_id: RuntimeId::new(),
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            Response::Loaded(_)
            | Response::Attached(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(SafeInitError::UnexpectedResponse),
        }
    }

    /// Load Catena programs from source paths.
    pub fn load<I>(&self, paths: I) -> Result<Artifact, SafeInitError>
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
        self.load_owned_sources(sources)
    }

    /// Load Catena programs from in-memory source strings.
    pub fn load_sources<'a, I>(&self, sources: I) -> Result<Artifact, SafeInitError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        self.load_owned_sources(sources.into_iter().map(ToOwned::to_owned).collect())
    }

    fn load_owned_sources(&self, sources: Vec<String>) -> Result<Artifact, SafeInitError> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| SafeInitError::Transport("worker lock was poisoned".to_string()))?;
        let mut worker = worker;
        worker
            .send(&Request::LoadSources { sources })
            .map_err(map_init_worker_error)?;

        match worker.receive().map_err(map_init_worker_error)? {
            Response::Loaded(Ok((index, entry_points))) => {
                Ok(Artifact::new(self.runtime_id, index, entry_points.into()))
            }
            Response::Loaded(Err(error)) => Err(SafeInitError::RemoteLoad(error)),
            Response::Initialized(_)
            | Response::Attached(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(SafeInitError::UnexpectedResponse),
        }
    }

    pub(crate) fn id(&self) -> RuntimeId {
        self.runtime_id
    }

    /// Attach one verified, read-only file to this worker session.
    pub(crate) fn attach_asset(
        &self,
        key: [u8; 32],
        file: File,
    ) -> Result<AttachedAsset, AssetError> {
        let byte_len = file.metadata().map_err(AssetError::FileMetadata)?.len();
        if validate_byte_len(byte_len).is_err() {
            return Err(AssetError::InvalidLength {
                actual: byte_len,
                maximum: MAX_ASSET_BYTES,
            });
        }

        let mut worker = self
            .worker
            .lock()
            .map_err(|_| AssetError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(AssetError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        worker
            .send(&Request::AttachAsset { key, byte_len })
            .map_err(map_asset_worker_error)?;
        worker.send_file(&file).map_err(map_asset_worker_error)?;

        match worker.receive().map_err(map_asset_worker_error)? {
            Response::Attached(Ok((id, reported_len))) if reported_len == byte_len => {
                Ok(AttachedAsset { id, byte_len })
            }
            Response::Attached(Ok(_)) => Err(AssetError::UnexpectedResponse),
            Response::Attached(Err(error)) => Err(AssetError::Remote(error)),
            Response::Initialized(_)
            | Response::Loaded(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(AssetError::UnexpectedResponse),
        }
    }

    pub(crate) fn bind_resident_model(
        &self,
        artifact: &Artifact,
        entry_point: String,
        assets: Vec<ResidentAssetSlice>,
        state_byte_multipliers: Vec<u64>,
        vocabulary_size: u64,
        maximum_capacity: u64,
    ) -> Result<ResidentModel, ResidentError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(ResidentError::Remote(
                "program belongs to a different GPU session".to_string(),
            ));
        }
        let binding = WireModelBinding {
            artifact: artifact.index(),
            entry_point,
            assets: assets
                .into_iter()
                .map(|slice| WireAssetSlice {
                    asset: slice.asset,
                    offset: slice.offset,
                    byte_len: slice.byte_len,
                })
                .collect(),
            state_byte_multipliers,
            vocabulary_size,
            maximum_capacity,
        };
        let mut worker = self.resident_worker()?;
        worker
            .send(&Request::BindModel { binding })
            .map_err(map_resident_worker_error)?;
        match worker.receive().map_err(map_resident_worker_error)? {
            Response::Resident(Ok(ResidentResponse::ModelBound(id))) => Ok(ResidentModel { id }),
            Response::Resident(Err(error)) => Err(ResidentError::Remote(error)),
            Response::Initialized(_)
            | Response::Loaded(_)
            | Response::Attached(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(ResidentError::UnexpectedResponse),
        }
    }

    pub(crate) fn start_resident_generation(
        &self,
        model: ResidentModel,
        capacity: u64,
    ) -> Result<ResidentGeneration, ResidentError> {
        let mut worker = self.resident_worker()?;
        worker
            .send(&Request::StartGeneration {
                model: model.id,
                capacity,
            })
            .map_err(map_resident_worker_error)?;
        match worker.receive().map_err(map_resident_worker_error)? {
            Response::Resident(Ok(ResidentResponse::GenerationStarted(id))) => {
                Ok(ResidentGeneration { id })
            }
            Response::Resident(Err(error)) => Err(ResidentError::Remote(error)),
            Response::Initialized(_)
            | Response::Loaded(_)
            | Response::Attached(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(ResidentError::UnexpectedResponse),
        }
    }

    pub(crate) fn step_resident_generation(
        &self,
        generation: ResidentGeneration,
        tokens: Vec<u32>,
    ) -> Result<u32, ResidentError> {
        let mut worker = self.resident_worker()?;
        worker
            .send(&Request::StepGeneration {
                generation: generation.id,
                tokens,
            })
            .map_err(map_resident_worker_error)?;
        match worker.receive().map_err(map_resident_worker_error)? {
            Response::Resident(Ok(ResidentResponse::Token(token))) => Ok(token),
            Response::Resident(Err(error)) => Err(ResidentError::Remote(error)),
            Response::Initialized(_)
            | Response::Loaded(_)
            | Response::Attached(_)
            | Response::Resident(_)
            | Response::Executed(_) => Err(ResidentError::UnexpectedResponse),
        }
    }

    pub(crate) fn release_resident_generation(&self, generation: ResidentGeneration) {
        if let Ok(mut worker) = self.resident_worker() {
            let _ = worker.send(&Request::ReleaseGeneration {
                generation: generation.id,
            });
        }
    }

    pub(crate) fn release_resident_model(&self, model: ResidentModel) {
        if let Ok(mut worker) = self.resident_worker() {
            let _ = worker.send(&Request::ReleaseModel { model: model.id });
        }
    }

    fn resident_worker(&self) -> Result<std::sync::MutexGuard<'_, WorkerProcess>, ResidentError> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| ResidentError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(ResidentError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        Ok(worker)
    }

    /// Run a source-level program in the child process.
    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], SafeExecError> {
        self.exec_values(artifact, name, args.into())?
            .try_into()
            .map_err(|_| SafeExecError::UnexpectedResponse)
    }

    /// Run a source-level program with dynamically sized inputs and outputs.
    pub fn exec_values<'a>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(SafeExecError::Runtime(ExecError::UnknownArtifact));
        }
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
                artifact: artifact.index(),
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
            Response::Initialized(_)
            | Response::Loaded(_)
            | Response::Attached(_)
            | Response::Resident(_) => {
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
    let asset_socket = unsafe { UnixStream::from_raw_fd(CHILD_ASSET_SOCKET_FD) };
    run_child_loop(stdin.lock(), stdout.lock(), &asset_socket)?;
    Ok(true)
}

fn run_child_loop(
    mut reader: impl Read,
    mut writer: impl io::Write,
    asset_socket: &UnixStream,
) -> Result<(), ChildMainError> {
    let request = read_request(&mut reader)?.ok_or(ChildMainError::ExpectedInitialization)?;
    let Request::Initialize { dialect } = request else {
        return Err(ChildMainError::ExpectedInitialization);
    };

    let mut runtime = match Runtime::new(dialect) {
        Ok(runtime) => runtime,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    let ipc = IpcTransport::from_runtime(&runtime);
    let mut assets = AssetStore::new(dialect);
    let mut resident = ResidentStore::new();
    write_response(&mut writer, &Response::Initialized(Ok(())))?;

    let mut pending_outputs = Vec::new();
    while let Some(request) = read_request(&mut reader)? {
        match request {
            Request::Initialize { .. } => return Err(ChildMainError::AlreadyInitialized),
            Request::LoadSources { sources } => {
                let result = runtime
                    .load_sources(sources.iter().map(String::as_str))
                    .map(|artifact| (artifact.index(), artifact.entry_points().to_vec()))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Loaded(result))?;
            }
            Request::AttachAsset { key, byte_len } => {
                let result = receive_file(asset_socket)
                    .map_err(|error| anyhow::anyhow!(error))
                    .and_then(|file| assets.attach(key, byte_len, file))
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::BindModel { binding } => {
                let result = resident
                    .bind_model(&runtime, &assets, binding)
                    .map(ResidentResponse::ModelBound)
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Resident(result))?;
            }
            Request::StartGeneration { model, capacity } => {
                let result = resident
                    .start_generation(&runtime, model, capacity)
                    .map(ResidentResponse::GenerationStarted)
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Resident(result))?;
            }
            Request::StepGeneration { generation, tokens } => {
                let result = match resident.step_generation(&runtime, &assets, generation, tokens) {
                    Ok(token) => Ok(ResidentResponse::Token(token)),
                    Err(error) => {
                        if let Some(error) = resident::gpu_synchronization(&error) {
                            return Err(ChildMainError::GpuSynchronization(error.to_string()));
                        }
                        Err(error.to_string())
                    }
                };
                write_response(&mut writer, &Response::Resident(result))?;
            }
            Request::ReleaseGeneration { generation } => {
                resident.release_generation(generation);
            }
            Request::ReleaseModel { model } => {
                resident.release_model(model);
            }
            Request::Shutdown => return Ok(()),
            Request::ReleaseOutputs => {
                pending_outputs.clear();
            }
            Request::Execute {
                artifact,
                name,
                buffers,
                args,
            } => {
                let response = if pending_outputs.is_empty() {
                    match runtime.artifact_at(artifact) {
                        Ok(artifact) => execute_in_child(
                            &runtime,
                            &ipc,
                            artifact,
                            &name,
                            buffers,
                            args,
                            &mut pending_outputs,
                        )?,
                        Err(error) => Response::Executed(Err(RemoteExecError::Runtime(error))),
                    }
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
    artifact: &Artifact,
    name: &str,
    buffers: Vec<WireIpcBuffer>,
    wire_args: Vec<WireValue>,
    pending_outputs: &mut Vec<MemOwn>,
) -> Result<Response, ChildMainError> {
    let imported = match import_ipc_buffers(ipc, buffers) {
        Ok(imported) => imported,
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Memory(error)))),
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
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Memory(error)))),
    };

    let values = match runtime.exec_values(artifact, name, args) {
        Ok(values) => values,
        Err(ExecError::GpuSynchronization(error)) => {
            return Err(ChildMainError::GpuSynchronization(error));
        }
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Runtime(error)))),
    };
    Ok(match encode_child_outputs(ipc, values, pending_outputs) {
        Ok(execution) => Response::Executed(Ok(execution)),
        Err(error) => {
            pending_outputs.clear();
            Response::Executed(Err(error))
        }
    })
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
    asset_socket: UnixStream,
    stderr_reader: Option<JoinHandle<Vec<u8>>>,
    termination: Option<Termination>,
}

#[derive(Debug, Error)]
enum WorkerError {
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error(transparent)]
    AssetDescriptor(#[from] FdTransportError),
    #[error("failed to wait for SafeRuntime child: {0}")]
    Wait(#[source] io::Error),
    #[error("SafeRuntime child terminated")]
    Terminated(Termination),
}

impl WorkerProcess {
    fn spawn(executable: &Path) -> Result<Self, SafeInitError> {
        let (asset_socket, child_asset_socket) =
            UnixStream::pair().map_err(SafeInitError::AssetTransport)?;
        let child_asset_fd = child_asset_socket.as_raw_fd();
        let mut command = Command::new(executable);
        command
            .env(CHILD_MODE_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(child_asset_fd, CHILD_ASSET_SOCKET_FD) == -1 {
                    return Err(io::Error::last_os_error());
                }
                if libc::fcntl(CHILD_ASSET_SOCKET_FD, libc::F_SETFD, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().map_err(|source| SafeInitError::Spawn {
            executable: executable.to_path_buf(),
            source,
        })?;
        drop(child_asset_socket);
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
            asset_socket,
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

    fn send_file(&mut self, file: &File) -> Result<(), WorkerError> {
        if let Err(error) = send_file(&self.asset_socket, file) {
            let _ = self.asset_socket.shutdown(Shutdown::Both);
            return Err(error.into());
        }
        Ok(())
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

fn map_asset_worker_error(error: WorkerError) -> AssetError {
    match error {
        WorkerError::Terminated(termination) => AssetError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => AssetError::Transport(other.to_string()),
    }
}

fn map_resident_worker_error(error: WorkerError) -> ResidentError {
    match error {
        WorkerError::Terminated(termination) => ResidentError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => ResidentError::Transport(other.to_string()),
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn structured_rejections_preserve_the_session() {
        assert!(!SafeInitError::RemoteLoad("invalid source".into()).invalidates_session());
        assert!(!AssetError::Remote("invalid asset".into()).invalidates_session());
        assert!(!ResidentError::Remote("invalid request".into()).invalidates_session());
    }

    #[test]
    fn protocol_failures_invalidate_the_session() {
        assert!(SafeInitError::UnexpectedResponse.invalidates_session());
        assert!(AssetError::Transport("closed".into()).invalidates_session());
        assert!(ResidentError::UnexpectedResponse.invalidates_session());
    }
}
