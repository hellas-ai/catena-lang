//! Process-isolated execution for Catena programs.
//!
//! Generated GPU code runs through native libraries, so a failed Catena assertion,
//! GPU runtime failure, or native crash can terminate the process rather than return
//! a Rust error. [`SafeRuntime`] solves this by executing [`Runtime`] in a child
//! process and communicating over a framed protocol. If execution terminates the
//! child, the host process survives and receives a structured error with its exit
//! status and stderr.

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    fs::File,
    hash::{Hash, Hasher},
    io::{self, BufReader, Read, Write},
    net::Shutdown,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    os::unix::{net::UnixStream, process::CommandExt},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
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
        FrameEncodeError, ProtocolError, RemoteExecError, Request, ResidentResponse, Response,
        WireAssetSlice, WireExecution, WireIpcBuffer, WireModelBinding, WireValue, encode_frame,
        read_frame, write_encoded_frame, write_frame,
    },
    resident::ResidentStore,
};

pub(crate) use self::assets::{MAX_RESIDENT_ASSET_BYTES, MAX_RESIDENT_ASSETS};
use crate::{
    codegen::GpuDialect,
    runtime::{
        Artifact as RuntimeArtifact, EntryPoint, ExecError, MemError, MemOwn, Runtime, RuntimeId,
        Value,
    },
};

const CHILD_MODE_ENV: &str = "CATENA_SAFE_RUNTIME_CHILD";
const CHILD_ASSET_SOCKET_FD: RawFd = 3;

/// Default wall-clock limit for one isolated source-load and GPU compilation.
pub const DEFAULT_COMPILE_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Default wall-clock limit for one isolated GPU operation or full generation.
pub const DEFAULT_EXECUTION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

const DROP_GRACE_TIMEOUT: Duration = Duration::from_millis(250);
const KILL_REAP_TIMEOUT: Duration = Duration::from_secs(1);
const STDERR_TAIL_BYTES: usize = 64 * 1024;

// One atomic word makes admission and retirement linearizable without putting
// a poisonable or fork-inherited mutex on SafeRuntime's process-wide spawn
// path. The high half identifies the process; the low half contains two u16
// counters (workers being spawned, then leaders awaiting a proven wait). The
// maximum retirement value is a sticky saturation sentinel, not a count.
//
// `spawning` is an admission claim, not something retirement waits for. Every
// constructor must mutate this word so it is totally ordered with a terminal
// transition. A constructor whose claim wins first may finish; retirement wins
// against every later constructor. Counting claims permits concurrent healthy
// session construction without serializing `Command::spawn`, while terminal
// paths stay bounded and never wait for those earlier operations.
const LIFECYCLE_PROCESS_SHIFT: u32 = 32;
const LIFECYCLE_RETIRING_SHIFT: u32 = 16;
const LIFECYCLE_COUNTER_MASK: u64 = u16::MAX as u64;
const LIFECYCLE_RETIREMENT_SATURATED: u16 = u16::MAX;

#[derive(Debug, Default)]
struct WorkerLifecycleGate {
    state: AtomicU64,
}

#[derive(Debug)]
struct WorkerSpawnPermit {
    gate: Arc<WorkerLifecycleGate>,
    process: u32,
}

#[derive(Debug)]
struct WorkerRetirementPermit {
    gate: Arc<WorkerLifecycleGate>,
    process: u32,
    completable: bool,
    completed: AtomicBool,
}

impl WorkerLifecycleGate {
    fn process_id() -> u32 {
        let process = unsafe { libc::getpid() };
        u32::try_from(process).expect("getpid must return a positive value representable as u32")
    }

    fn encode(process: u32, spawning: u16, retiring: u16) -> u64 {
        (u64::from(process) << LIFECYCLE_PROCESS_SHIFT)
            | (u64::from(retiring) << LIFECYCLE_RETIRING_SHIFT)
            | u64::from(spawning)
    }

    fn decode(state: u64) -> (u32, u16, u16) {
        let process = (state >> LIFECYCLE_PROCESS_SHIFT) as u32;
        let retiring = ((state >> LIFECYCLE_RETIRING_SHIFT) & LIFECYCLE_COUNTER_MASK) as u16;
        let spawning = (state & LIFECYCLE_COUNTER_MASK) as u16;
        (process, spawning, retiring)
    }

    fn begin_spawn(self: &Arc<Self>) -> Result<WorkerSpawnPermit, SafeInitError> {
        self.begin_spawn_for(Self::process_id())
    }

    fn begin_spawn_for(self: &Arc<Self>, process: u32) -> Result<WorkerSpawnPermit, SafeInitError> {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            let next = if owner != process {
                // A post-fork child has no ownership over the parent's worker
                // leaders. Reset the copied counters before its first spawn.
                Self::encode(process, 1, 0)
            } else {
                if retiring != 0 {
                    return Err(SafeInitError::PreviousWorkerUnreaped { workers: retiring });
                }
                let Some(spawning) = spawning.checked_add(1) else {
                    return Err(SafeInitError::LifecycleCapacity);
                };
                Self::encode(process, spawning, retiring)
            };
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(WorkerSpawnPermit {
                    gate: self.clone(),
                    process,
                });
            }
        }
    }

    fn begin_retirement(self: &Arc<Self>) -> WorkerRetirementPermit {
        self.begin_retirement_for(Self::process_id())
    }

    fn begin_retirement_for(self: &Arc<Self>, process: u32) -> WorkerRetirementPermit {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            let (next, completable) = if owner != process {
                (Self::encode(process, 0, 1), true)
            } else {
                // Reserve u16::MAX as a sticky saturation sentinel. Once an
                // additional leader cannot be counted, no tracked completion
                // may reopen admission while that untracked leader can remain.
                if retiring >= LIFECYCLE_RETIREMENT_SATURATED - 1 {
                    (
                        Self::encode(process, spawning, LIFECYCLE_RETIREMENT_SATURATED),
                        false,
                    )
                } else {
                    (Self::encode(process, spawning, retiring + 1), true)
                }
            };
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return WorkerRetirementPermit {
                    gate: self.clone(),
                    process,
                    completable,
                    completed: AtomicBool::new(false),
                };
            }
        }
    }

    fn finish_spawn(&self, process: u32) {
        self.update_current_process(process, |spawning, retiring| {
            (spawning.saturating_sub(1), retiring)
        });
    }

    fn finish_retirement(&self, process: u32) {
        self.update_current_process(process, |spawning, retiring| {
            let retiring = if retiring == LIFECYCLE_RETIREMENT_SATURATED {
                LIFECYCLE_RETIREMENT_SATURATED
            } else {
                retiring.saturating_sub(1)
            };
            (spawning, retiring)
        });
    }

    fn update_current_process(&self, process: u32, update: impl Fn(u16, u16) -> (u16, u16)) {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            if owner != process {
                // The permit was copied across fork. It belongs to the parent
                // epoch and must not alter the child's freshly reset counters.
                return;
            }
            let (spawning, retiring) = update(spawning, retiring);
            let next = Self::encode(process, spawning, retiring);
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

impl Drop for WorkerSpawnPermit {
    fn drop(&mut self) {
        self.gate.finish_spawn(self.process);
    }
}

impl WorkerRetirementPermit {
    /// Complete only after `wait(2)` has positively established leader exit.
    /// Merely dropping this value is deliberately fail-closed.
    fn finish_after_wait(&self) {
        if self.completable && !self.completed.swap(true, Ordering::AcqRel) {
            self.gate.finish_retirement(self.process);
        }
    }
}

fn worker_lifecycle_gate() -> Arc<WorkerLifecycleGate> {
    static GATE: std::sync::OnceLock<Arc<WorkerLifecycleGate>> = std::sync::OnceLock::new();
    GATE.get_or_init(|| Arc::new(WorkerLifecycleGate::default()))
        .clone()
}

/// Deadlines applied by a process-isolated runtime.
///
/// The compile deadline covers each complete source request. The execution
/// deadline covers initialization, asset and model control requests, generic
/// execution, and one complete resident generation from its start request
/// through its final release. It is deliberately not restarted per token.
/// Operations first wait to acquire the session lease; that queue wait happens
/// before the corresponding deadline is armed. A resident generation retains
/// the lease and its watchdog from start through release, including synchronous
/// host callbacks and inter-step gaps. Callback code cannot be preempted or
/// bounded by the watchdog and must not re-enter the same session; re-entry is
/// rejected instead of waiting on itself. If a callback never returns after
/// the watchdog kills its worker, the parent cannot reap that worker and
/// process-wide worker admission cannot reopen unless the callback returns and
/// teardown subsequently proves leader exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeRuntimeTimeouts {
    compile: Duration,
    execution: Duration,
}

impl SafeRuntimeTimeouts {
    /// Start from the production defaults.
    pub const fn new() -> Self {
        Self {
            compile: DEFAULT_COMPILE_TIMEOUT,
            execution: DEFAULT_EXECUTION_TIMEOUT,
        }
    }

    /// Set the deadline for one complete source-load and GPU-compile request.
    pub const fn with_compile_timeout(mut self, timeout: Duration) -> Self {
        self.compile = timeout;
        self
    }

    /// Set the deadline for one control/execution operation or full generation.
    pub const fn with_execution_timeout(mut self, timeout: Duration) -> Self {
        self.execution = timeout;
        self
    }

    /// Return the deadline for one source-load and compilation request.
    #[must_use]
    pub const fn compile_timeout(self) -> Duration {
        self.compile
    }

    /// Return the deadline for one non-compilation operation or full generation.
    #[must_use]
    pub const fn execution_timeout(self) -> Duration {
        self.execution
    }
}

impl Default for SafeRuntimeTimeouts {
    fn default() -> Self {
        Self::new()
    }
}

/// Failures while attaching or slicing a session-resident asset.
#[derive(Debug, Error)]
pub enum AssetError {
    #[error("failed to inspect verified asset descriptor: {0}")]
    FileMetadata(#[source] io::Error),
    #[error("asset length must be between 1 and {maximum} bytes, got {actual}")]
    InvalidLength { actual: u64, maximum: u64 },
    #[error("asset transport failed: {0}")]
    Transport(String),
    #[error("invalid local asset request: {0}")]
    InvalidRequest(String),
    #[error("SafeRuntime asset attachment exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child rejected asset: {0}")]
    Remote(String),
    #[error("SafeRuntime child returned an unexpected asset response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated while attaching an asset with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error("asset belongs to a different GPU asset owner")]
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
    /// Caller-side validation, local request preparation, and a structured
    /// rejection from the worker leave the session usable. Transport failures,
    /// termination, and an unexpected response do not: the next operation must
    /// use a new session.
    pub fn invalidates_session(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::TimedOut { .. }
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
    #[error("invalid local resident GPU request: {0}")]
    InvalidRequest(String),
    #[error("resident GPU teardown failed: {0}")]
    Teardown(String),
    #[error("SafeRuntime child returned an unexpected resident response")]
    UnexpectedResponse,
    #[error("resident GPU transport failed: {0}")]
    Transport(String),
    #[error("resident GPU operation exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child terminated during a resident GPU request with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
}

impl ResidentError {
    /// Whether the worker protocol can no longer be trusted after this error.
    pub fn invalidates_session(&self) -> bool {
        !matches!(self, Self::Remote(_) | Self::InvalidRequest(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentModel {
    pub(crate) id: u64,
}

#[derive(Debug)]
pub(crate) struct ResidentGeneration {
    pub(crate) id: u64,
    deadline: ProcessGroupDeadline,
    timeout: Duration,
    _operation: SessionOperationLease,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentAssetSlice {
    pub(crate) asset: u64,
    pub(crate) offset: u64,
    pub(crate) byte_len: u64,
}

#[derive(Debug)]
pub(crate) struct ResidentModelBinding {
    pub(crate) entry_point: String,
    pub(crate) assets: Vec<ResidentAssetSlice>,
    pub(crate) state_byte_multipliers: Vec<u64>,
    pub(crate) vocabulary_size: u64,
    pub(crate) maximum_capacity: u64,
    pub(crate) generation_device_allocation_budget_bytes: u64,
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
    #[error("failed to create SafeRuntime worker scratch directory under {parent}: {source}")]
    Scratch {
        parent: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to create SafeRuntime asset transport: {0}")]
    AssetTransport(#[source] io::Error),
    #[error("SafeRuntime setup transport failed: {0}")]
    Transport(String),
    #[error("invalid local SafeRuntime setup request: {0}")]
    InvalidRequest(String),
    #[error("failed to inspect the host SIGCHLD disposition: {0}")]
    ChildSignal(#[source] io::Error),
    #[error(
        "SafeRuntime requires waitable child processes; SIGCHLD must not use SIG_IGN or SA_NOCLDWAIT"
    )]
    UnwaitableChildren,
    #[error("failed to start a SafeRuntime lifecycle support thread: {0}")]
    LifecycleThread(#[source] io::Error),
    #[error("cannot start SafeRuntime while {workers} previous worker leader(s) remain unreaped")]
    PreviousWorkerUnreaped { workers: u16 },
    #[error("SafeRuntime process lifecycle tracker exhausted its concurrent-spawn capacity")]
    LifecycleCapacity,
    #[error("SafeRuntime child initialization failed: {0}")]
    RemoteInitialization(String),
    #[error("SafeRuntime child failed to load sources: {0}")]
    RemoteLoad(String),
    #[error("SafeRuntime source loading and GPU compilation exceeded {timeout:?}")]
    CompileTimedOut { timeout: Duration },
    #[error("SafeRuntime initialization exceeded {timeout:?}")]
    InitializationTimedOut { timeout: Duration },
    #[error("SafeRuntime compile timeout must be greater than zero")]
    InvalidCompileTimeout,
    #[error("SafeRuntime execution timeout must be greater than zero")]
    InvalidExecutionTimeout,
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
    /// Of these variants, only source reading, local request preparation, and
    /// a structured load rejection can occur without invalidating a live
    /// session. The construction-only variants are conservatively classified
    /// as invalidating so callers never retain a session whose health is
    /// uncertain.
    pub fn invalidates_session(&self) -> bool {
        !matches!(
            self,
            Self::ReadSource { .. } | Self::RemoteLoad(_) | Self::InvalidRequest(_)
        )
    }
}

/// Execution failures reported by [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeExecError {
    #[error("SafeRuntime artifact is no longer loaded in its child")]
    UnknownArtifact,
    #[error(transparent)]
    Runtime(#[from] ExecError),
    #[error("SafeRuntime transport failed: {0}")]
    Transport(String),
    #[error("invalid local SafeRuntime execution request: {0}")]
    InvalidRequest(String),
    #[error("SafeRuntime child rejected memory IPC: {0}")]
    RemoteMemory(String),
    #[error("SafeRuntime execution exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child returned an unexpected execution response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] MemError),
}

impl SafeExecError {
    /// Whether the worker protocol can no longer be trusted after this error.
    ///
    /// Local validation/allocation errors and structured runtime or child IPC
    /// rejections leave request/response synchronization intact. Transport
    /// failures, deadlines, termination, and semantically impossible responses
    /// require a new session.
    pub fn invalidates_session(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::TimedOut { .. }
                | Self::UnexpectedResponse
                | Self::ChildTerminated { .. }
                | Self::Unavailable { .. }
        )
    }
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

/// Serializes every operation that can touch one worker session. A generation
/// keeps its lease across host callbacks and between protocol steps, so its
/// still-armed watchdog can never kill the worker while an unrelated operation
/// is using it.
#[derive(Debug, Default)]
struct SessionOperationGate {
    state: Mutex<SessionOperationState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct SessionOperationState {
    active: Option<ActiveSessionOperation>,
    model_release_deferrals: ModelReleaseDeferrals,
    #[cfg(test)]
    model_release_waiters: usize,
}

#[derive(Debug, Default)]
enum ModelReleaseDeferrals {
    #[default]
    Closed,
    Open(Vec<DeferredRelease>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeferredRelease {
    Model(u64),
    Artifact(usize),
}

impl DeferredRelease {
    fn request(self) -> Request {
        match self {
            Self::Model(model) => Request::ReleaseModel { model },
            Self::Artifact(artifact) => Request::ReleaseArtifact { artifact },
        }
    }

    fn acknowledged(self, response: &Response) -> bool {
        match (self, response) {
            (
                Self::Model(expected),
                Response::Resident(Ok(ResidentResponse::ModelReleased(id))),
            ) => expected == *id,
            (
                Self::Artifact(expected),
                Response::Resident(Ok(ResidentResponse::ArtifactReleased(id))),
            ) => expected == *id,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveSessionOperation {
    owner: thread::ThreadId,
    kind: SessionOperationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionOperationKind {
    Ordinary,
    Generation,
}

#[derive(Debug)]
struct SessionOperationLease {
    gate: Arc<SessionOperationGate>,
    operation: ActiveSessionOperation,
}

impl SessionOperationGate {
    fn acquire(self: &Arc<Self>, kind: SessionOperationKind) -> Result<SessionOperationLease, ()> {
        let operation = ActiveSessionOperation {
            owner: thread::current().id(),
            kind,
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match state.active {
                None => {
                    debug_assert!(matches!(
                        &state.model_release_deferrals,
                        ModelReleaseDeferrals::Closed
                    ));
                    state.active = Some(operation);
                    return Ok(SessionOperationLease {
                        gate: self.clone(),
                        operation,
                    });
                }
                Some(active) if active.owner == operation.owner => return Err(()),
                Some(_) => {
                    state = self
                        .available
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        }
    }

    /// Admit one model destructor without racing generation startup. A release
    /// waits while startup is uncommitted, then either joins the generation's
    /// open deferral batch or acquires the ordinary lease after startup rolls
    /// back. Same-thread non-generation re-entry remains an error.
    #[cfg(test)]
    fn acquire_or_defer_model_release(
        self: &Arc<Self>,
        model: u64,
    ) -> Result<Option<SessionOperationLease>, ()> {
        self.acquire_or_defer_release(DeferredRelease::Model(model))
    }

    fn acquire_or_defer_release(
        self: &Arc<Self>,
        release: DeferredRelease,
    ) -> Result<Option<SessionOperationLease>, ()> {
        let operation = ActiveSessionOperation {
            owner: thread::current().id(),
            kind: SessionOperationKind::Ordinary,
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match state.active {
                None => {
                    debug_assert!(matches!(
                        &state.model_release_deferrals,
                        ModelReleaseDeferrals::Closed
                    ));
                    state.active = Some(operation);
                    return Ok(Some(SessionOperationLease {
                        gate: self.clone(),
                        operation,
                    }));
                }
                Some(active) if active.kind == SessionOperationKind::Generation => {
                    if let ModelReleaseDeferrals::Open(releases) =
                        &mut state.model_release_deferrals
                    {
                        releases.push(release);
                        return Ok(None);
                    }
                    if active.owner == operation.owner {
                        return Err(());
                    }
                }
                Some(active) if active.owner == operation.owner => return Err(()),
                Some(_) => {}
            }
            #[cfg(test)]
            {
                state.model_release_waiters += 1;
                self.available.notify_all();
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            #[cfg(test)]
            {
                debug_assert!(state.model_release_waiters != 0);
                state.model_release_waiters -= 1;
            }
        }
    }

    #[cfg(test)]
    fn wait_for_model_release_waiter(&self) {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.model_release_waiters == 0 {
            let now = Instant::now();
            assert!(now < deadline, "model release did not wait on the gate");
            let (next, result) = self
                .available
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            assert!(
                !result.timed_out() || state.model_release_waiters != 0,
                "model release did not wait on the gate"
            );
        }
    }

    /// Commit a successfully started generation to accepting model-release
    /// deferrals, then wake every destructor that waited during worker I/O.
    fn open_model_release_deferrals(&self, operation: ActiveSessionOperation) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(operation));
        debug_assert_eq!(operation.kind, SessionOperationKind::Generation);
        if matches!(
            &state.model_release_deferrals,
            ModelReleaseDeferrals::Closed
        ) {
            state.model_release_deferrals = ModelReleaseDeferrals::Open(Vec::new());
            self.available.notify_all();
        } else {
            debug_assert!(false, "generation model-release deferral was already open");
        }
    }

    /// Atomically close deferral and take every release accepted before this
    /// point. A concurrent destructor is therefore either in this returned
    /// batch or observes `Closed` and waits for the ordinary operation lease.
    fn seal_model_release_deferrals(
        &self,
        operation: ActiveSessionOperation,
    ) -> Vec<DeferredRelease> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(operation));
        debug_assert_eq!(operation.kind, SessionOperationKind::Generation);
        match std::mem::take(&mut state.model_release_deferrals) {
            ModelReleaseDeferrals::Open(releases) => releases,
            ModelReleaseDeferrals::Closed => {
                debug_assert!(
                    false,
                    "generation model-release deferral was already sealed"
                );
                Vec::new()
            }
        }
    }
}

impl SessionOperationLease {
    fn open_model_release_deferrals(&self) {
        self.gate.open_model_release_deferrals(self.operation);
    }

    fn seal_model_release_deferrals(&self) -> Vec<DeferredRelease> {
        self.gate.seal_model_release_deferrals(self.operation)
    }
}

impl Drop for SessionOperationLease {
    fn drop(&mut self) {
        let mut state = self
            .gate
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(self.operation));
        // A missing protocol teardown already invalidates the worker. Never
        // let stale destructor work escape into the next operation's lease.
        state.model_release_deferrals = ModelReleaseDeferrals::Closed;
        state.active = None;
        self.gate.available.notify_all();
    }
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
///
/// The host must also leave `SIGCHLD` waitable for the session lifetime and
/// must not reap children it did not spawn. Construction rejects `SIG_IGN` and
/// `SA_NOCLDWAIT`; external `waitpid(-1, ...)` use would violate the exclusive
/// ownership required by [`std::process::Child`] and the worker's process-group
/// guard.
///
/// Worker-retirement admission is process-local. A changed PID resets copied
/// admission counters so an exec-style fork child does not inherit its parent's
/// retired leaders. Live `SafeRuntime` values themselves are not fork-safe:
/// after a multi-threaded fork, the child must exec rather than use or drop
/// inherited sessions and synchronization primitives. Independently launched
/// host processes require a deployment-level memory/cgroup boundary.
///
/// The worker intentionally inherits the host environment except for `TMPDIR`,
/// which is replaced by a private, parent-owned scratch directory. GPU runtime
/// and compiler discovery currently depend on platform-specific variables such
/// as `PATH`, `ROCM_PATH`, `HIP_PATH`, `CUDA_PATH`, and loader/wrapper settings;
/// there is no portable safe allowlist at this layer. Deployments that require
/// environment isolation must start the host process with a curated environment
/// rather than assuming `SafeRuntime` clears it.
///
/// This lower-level interface supports raw [`Value`] memory across the process
/// boundary, so its public constructors initialize GPU IPC in the parent. The
/// typed [`crate::safe_gpu::Session`] interface keeps all device memory in the
/// worker and therefore loads the native GPU runtime only there.
#[derive(Debug, Clone)]
pub struct SafeRuntime {
    dialect: GpuDialect,
    worker: Arc<Mutex<WorkerProcess>>,
    asset_owner: Option<Arc<SafeRuntime>>,
    operation_gate: Arc<SessionOperationGate>,
    parent_ipc: Option<IpcTransport>,
    runtime_id: RuntimeId,
    timeouts: SafeRuntimeTimeouts,
}

/// A compiled program that retains its isolated worker after `SafeRuntime`
/// is dropped. Its final clone releases child-side artifact state; models that
/// were already bound retain their own reference to the executable.
#[derive(Debug, Clone)]
pub struct Artifact {
    inner: Arc<RemoteArtifact>,
}

#[derive(Debug)]
struct RemoteArtifact {
    runtime: SafeRuntime,
    index: usize,
    entry_points: Arc<[EntryPoint]>,
}

impl Artifact {
    fn new(runtime: SafeRuntime, index: usize, entry_points: Arc<[EntryPoint]>) -> Self {
        Self {
            inner: Arc::new(RemoteArtifact {
                runtime,
                index,
                entry_points,
            }),
        }
    }

    pub(crate) fn belongs_to(&self, runtime: RuntimeId) -> bool {
        self.inner.runtime.runtime_id == runtime
    }

    pub(crate) fn index(&self) -> usize {
        self.inner.index
    }

    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.inner.entry_points
    }

    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], SafeExecError> {
        let values = self.exec_values(name, args.into())?;
        let actual = values.len();
        values.try_into().map_err(|_| {
            SafeExecError::InvalidRequest(format!(
                "caller expected {N} outputs, but the entry point returned {actual}"
            ))
        })
    }

    pub fn exec_values<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        self.inner.runtime.exec_values(self, name, args)
    }
}

impl PartialEq for Artifact {
    fn eq(&self, other: &Self) -> bool {
        self.inner.runtime.runtime_id == other.inner.runtime.runtime_id
            && self.index() == other.index()
    }
}
impl Eq for Artifact {}
impl Hash for Artifact {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.runtime.runtime_id.hash(state);
        self.index().hash(state);
    }
}
impl Drop for RemoteArtifact {
    fn drop(&mut self) {
        // Release failures invalidate the worker, as with model teardown.
        let _ = self
            .runtime
            .release_object(DeferredRelease::Artifact(self.index));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParentInterface {
    RawValues,
    ResidentProtocol,
    AssetOwner,
}

fn initialize_parent_ipc<T, E>(
    interface: ParentInterface,
    load: impl FnOnce() -> Result<T, E>,
) -> Result<Option<T>, E> {
    match interface {
        ParentInterface::RawValues => load().map(Some),
        ParentInterface::ResidentProtocol | ParentInterface::AssetOwner => Ok(None),
    }
}

impl SafeRuntime {
    /// Construct an empty process-isolated runtime.
    pub fn new(dialect: GpuDialect) -> Result<Self, SafeInitError> {
        Self::with_timeouts(dialect, SafeRuntimeTimeouts::default())
    }

    /// Construct a runtime whose complete source-load and GPU-compile step is
    /// bounded by `compile_timeout`.
    ///
    /// A timeout kills the isolated worker and its compiler process group. The
    /// runtime must not be reused after [`SafeInitError::CompileTimedOut`].
    pub fn with_compile_timeout(
        dialect: GpuDialect,
        compile_timeout: Duration,
    ) -> Result<Self, SafeInitError> {
        Self::with_timeouts(
            dialect,
            SafeRuntimeTimeouts::default().with_compile_timeout(compile_timeout),
        )
    }

    /// Construct a runtime with provider-selected compile and execution limits.
    pub fn with_timeouts(
        dialect: GpuDialect,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_interface(dialect.into(), timeouts, ParentInterface::RawValues)
    }

    /// Construct the worker-resident protocol used by `safe_gpu` without
    /// loading GPU IPC or native GPU libraries in the parent process.
    pub(crate) fn with_resident_timeouts(
        dialect: GpuDialect,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_resident_backend(dialect.into(), timeouts)
    }

    pub(crate) fn with_resident_backend(
        backend: crate::runtime::Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        let owner = Arc::new(Self::with_asset_owner(backend, timeouts)?);
        Self::with_shared_assets(owner, timeouts)
    }

    pub(crate) fn with_asset_owner(
        backend: crate::runtime::Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_interface(backend, timeouts, ParentInterface::AssetOwner)
    }

    pub(crate) fn with_shared_assets(
        owner: Arc<Self>,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        let mut runtime = Self::with_interface(
            owner.dialect.into(),
            timeouts,
            ParentInterface::ResidentProtocol,
        )?;
        runtime.asset_owner = Some(owner);
        Ok(runtime)
    }

    pub(crate) fn asset_owner_id(&self) -> RuntimeId {
        self.asset_owner
            .as_ref()
            .map_or(self.runtime_id, |owner| owner.runtime_id)
    }

    pub(crate) fn asset_owner_is_available(&self) -> bool {
        let owner = self.asset_owner.as_deref().unwrap_or(self);
        let Ok(worker) = owner.worker.lock() else {
            return false;
        };
        if worker.ensure_available().is_err() {
            return false;
        }
        let Some(child) = worker.child.as_ref() else {
            return false;
        };
        // Observe without reaping: the lifecycle gate must retain the numeric
        // process-group identity until terminal cleanup has killed descendants.
        let mut status = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        unsafe {
            libc::waitid(
                libc::P_PID,
                child.id(),
                status.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            ) == 0
                && status.assume_init().si_pid() == 0
        }
    }

    fn with_interface(
        backend: crate::runtime::Backend,
        timeouts: SafeRuntimeTimeouts,
        interface: ParentInterface,
    ) -> Result<Self, SafeInitError> {
        if timeouts.compile_timeout().is_zero() {
            return Err(SafeInitError::InvalidCompileTimeout);
        }
        if timeouts.execution_timeout().is_zero() {
            return Err(SafeInitError::InvalidExecutionTimeout);
        }
        ensure_waitable_children()?;
        let executable = env::current_exe().map_err(SafeInitError::CurrentExecutable)?;
        let parent_ipc = initialize_parent_ipc(interface, || {
            IpcTransport::load(
                backend
                    .dialect()
                    .expect("raw-value runtime requires an explicit dialect"),
            )
        })?;
        let mut worker = WorkerProcess::spawn(&executable)?;

        match worker
            .with_deadline(
                timeouts.execution_timeout(),
                DeadlineKind::Initialize,
                |worker, deadline| {
                    worker.request(
                        &if interface == ParentInterface::AssetOwner {
                            Request::InitializeAssets { backend }
                        } else {
                            Request::Initialize { backend }
                        },
                        deadline,
                    )
                },
            )
            .map_err(map_init_worker_error)?
        {
            Response::Initialized(Ok(dialect))
                if backend
                    .dialect()
                    .is_some_and(|expected| expected != dialect) =>
            {
                Err(SafeInitError::UnexpectedResponse)
            }
            Response::Initialized(Ok(dialect)) => Ok(Self {
                dialect,
                worker: Arc::new(Mutex::new(worker)),
                asset_owner: None,
                operation_gate: Arc::new(SessionOperationGate::default()),
                parent_ipc,
                runtime_id: RuntimeId::new(),
                timeouts,
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            _ => Err(SafeInitError::UnexpectedResponse),
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
        let _operation = self.begin_operation().map_err(map_init_worker_error)?;
        let worker = self
            .worker
            .lock()
            .map_err(|_| SafeInitError::Transport("worker lock was poisoned".to_string()))?;
        let mut worker = worker;
        match worker
            .with_deadline(
                self.timeouts.compile_timeout(),
                DeadlineKind::Compile,
                |worker, deadline| worker.request(&Request::LoadSources { sources }, deadline),
            )
            .map_err(map_init_worker_error)?
        {
            Response::Loaded(Ok((index, entry_points))) => {
                Ok(Artifact::new(self.clone(), index, entry_points.into()))
            }
            Response::Loaded(Err(error)) => Err(SafeInitError::RemoteLoad(error)),
            _ => Err(worker.reject_protocol_response(SafeInitError::UnexpectedResponse)),
        }
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub(crate) fn id(&self) -> RuntimeId {
        self.runtime_id
    }

    fn begin_operation(&self) -> Result<SessionOperationLease, WorkerError> {
        self.operation_gate
            .acquire(SessionOperationKind::Ordinary)
            .map_err(|()| WorkerError::ReentrantOperation)
    }

    fn begin_generation(&self) -> Result<SessionOperationLease, WorkerError> {
        self.operation_gate
            .acquire(SessionOperationKind::Generation)
            .map_err(|()| WorkerError::ReentrantOperation)
    }

    /// Attach one verified, read-only file to this worker session.
    pub(crate) fn attach_asset(
        &self,
        key: [u8; 32],
        file: File,
    ) -> Result<AttachedAsset, AssetError> {
        if let Some(owner) = &self.asset_owner {
            return owner.attach_asset(key, file);
        }
        let byte_len = file.metadata().map_err(AssetError::FileMetadata)?.len();
        if validate_byte_len(byte_len).is_err() {
            return Err(AssetError::InvalidLength {
                actual: byte_len,
                maximum: MAX_ASSET_BYTES,
            });
        }

        let _operation = self.begin_operation().map_err(map_asset_worker_error)?;
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
        match worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    worker.send(&Request::AttachAsset { key, byte_len }, deadline)?;
                    worker.send_file(&file)?;
                    worker.receive(deadline)
                },
            )
            .map_err(map_asset_worker_error)?
        {
            Response::Attached(Ok((id, reported_len))) if reported_len == byte_len => {
                Ok(AttachedAsset { id, byte_len })
            }
            Response::Attached(Ok(_)) => {
                Err(worker.reject_protocol_response(AssetError::UnexpectedResponse))
            }
            Response::Attached(Err(error)) => Err(AssetError::Remote(error)),
            _ => Err(worker.reject_protocol_response(AssetError::UnexpectedResponse)),
        }
    }

    pub(crate) fn asset_usage(&self) -> Result<(u64, u64, u64), AssetError> {
        if let Some(owner) = &self.asset_owner {
            return owner.asset_usage();
        }
        let _operation = self.begin_operation().map_err(map_asset_worker_error)?;
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| AssetError::Transport("worker lock was poisoned".into()))?;
        match worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| worker.request(&Request::AssetUsage, deadline),
            )
            .map_err(map_asset_worker_error)?
        {
            Response::AssetUsage(count, uploaded, device) => Ok((count, uploaded, device)),
            _ => Err(worker.reject_protocol_response(AssetError::UnexpectedResponse)),
        }
    }

    fn export_asset(&self, asset: u64) -> Result<(File, u64, u64), ResidentError> {
        let _operation = self.begin_operation().map_err(map_resident_worker_error)?;
        let mut worker = self.resident_worker()?;
        worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    let response = worker.request(&Request::ExportAsset { asset }, deadline)?;
                    let result = match response {
                        Response::Exported(Ok((byte_len, allocation_len))) => {
                            deadline
                                .wait_for_descriptor(worker.asset_socket.as_raw_fd(), libc::POLLIN)
                                .map_err(ProtocolError::Io)?;
                            let file = receive_file(&worker.asset_socket)?;
                            Ok((file, byte_len, allocation_len))
                        }
                        Response::Exported(Err(error)) => Err(ResidentError::Remote(error)),
                        _ => {
                            Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse))
                        }
                    };
                    Ok(result)
                },
            )
            .map_err(map_resident_worker_error)?
    }

    pub(crate) fn bind_resident_model(
        &self,
        artifact: &Artifact,
        binding: ResidentModelBinding,
    ) -> Result<ResidentModel, ResidentError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(ResidentError::Remote(
                "program belongs to a different GPU session".to_string(),
            ));
        }
        let binding = WireModelBinding {
            artifact: artifact.index(),
            entry_point: binding.entry_point,
            assets: binding
                .assets
                .into_iter()
                .map(|slice| WireAssetSlice {
                    asset: slice.asset,
                    offset: slice.offset,
                    byte_len: slice.byte_len,
                })
                .collect(),
            state_byte_multipliers: binding.state_byte_multipliers,
            vocabulary_size: binding.vocabulary_size,
            maximum_capacity: binding.maximum_capacity,
            generation_device_allocation_budget_bytes: binding
                .generation_device_allocation_budget_bytes,
        };
        let _operation = self.begin_operation().map_err(map_resident_worker_error)?;
        let mut worker = self.resident_worker()?;
        for slice in &binding.assets {
            if worker.imported_assets.contains(&slice.asset) {
                continue;
            }
            let owner = self
                .asset_owner
                .as_ref()
                .ok_or_else(|| ResidentError::Remote("session has no asset owner".into()))?;
            let (file, byte_len, allocation_len) = owner.export_asset(slice.asset)?;
            let response = worker
                .with_deadline(
                    self.timeouts.execution_timeout(),
                    DeadlineKind::Execution,
                    |worker, deadline| {
                        worker.send(
                            &Request::ImportAsset {
                                asset: slice.asset,
                                byte_len,
                                allocation_len,
                            },
                            deadline,
                        )?;
                        worker.send_file(&file)?;
                        worker.receive(deadline)
                    },
                )
                .map_err(map_resident_worker_error)?;
            match response {
                Response::Attached(Ok((id, len))) if id == slice.asset && len == byte_len => {
                    worker.imported_assets.insert(id);
                }
                Response::Attached(Err(error)) => return Err(ResidentError::Remote(error)),
                _ => return Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse)),
            }
        }
        match worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| worker.request(&Request::BindModel { binding }, deadline),
            )
            .map_err(map_resident_worker_error)?
        {
            Response::Resident(Ok(ResidentResponse::ModelBound(id))) => Ok(ResidentModel { id }),
            Response::Resident(Err(error)) => Err(ResidentError::Remote(error)),
            _ => Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse)),
        }
    }

    pub(crate) fn start_resident_generation(
        &self,
        model: ResidentModel,
        capacity: u64,
    ) -> Result<ResidentGeneration, ResidentError> {
        let operation = self.begin_generation().map_err(map_resident_worker_error)?;
        let mut worker = self.resident_worker()?;
        let timeout = self.timeouts.execution_timeout();
        let deadline = worker
            .arm_deadline(timeout)
            .map_err(map_resident_worker_error)?;
        let response = worker.under_deadline(
            &deadline,
            timeout,
            DeadlineKind::Execution,
            |worker, deadline| {
                worker.request(
                    &Request::StartGeneration {
                        model: model.id,
                        capacity,
                    },
                    deadline,
                )
            },
        );
        match response {
            Ok(Response::Resident(Ok(ResidentResponse::GenerationStarted(id)))) => {
                // Only the worker acknowledgement commits generation startup.
                // Wake model destructors that waited rather than risking a
                // queued release being discarded by a healthy start failure.
                operation.open_model_release_deferrals();
                Ok(ResidentGeneration {
                    id,
                    deadline,
                    timeout,
                    _operation: operation,
                })
            }
            Ok(Response::Resident(Err(error))) => {
                worker
                    .finish_deadline(deadline, timeout, DeadlineKind::Execution)
                    .map_err(map_resident_worker_error)?;
                Err(ResidentError::Remote(error))
            }
            Ok(_) => {
                worker
                    .finish_deadline(deadline, timeout, DeadlineKind::Execution)
                    .map_err(map_resident_worker_error)?;
                Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse))
            }
            Err(error) => {
                worker
                    .finish_deadline(deadline, timeout, DeadlineKind::Execution)
                    .map_err(map_resident_worker_error)?;
                Err(map_resident_worker_error(error))
            }
        }
    }

    pub(crate) fn step_resident_generation(
        &self,
        generation: &ResidentGeneration,
        tokens: Vec<u32>,
    ) -> Result<u32, ResidentError> {
        let mut worker = self.resident_worker()?;
        match worker
            .under_deadline(
                &generation.deadline,
                generation.timeout,
                DeadlineKind::Execution,
                |worker, deadline| {
                    worker.request(
                        &Request::StepGeneration {
                            generation: generation.id,
                            tokens,
                        },
                        deadline,
                    )
                },
            )
            .map_err(map_resident_worker_error)?
        {
            Response::Resident(Ok(ResidentResponse::Token(token))) => Ok(token),
            Response::Resident(Err(error)) => Err(ResidentError::Remote(error)),
            _ => Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse)),
        }
    }

    pub(crate) fn release_resident_generation(
        &self,
        generation: ResidentGeneration,
    ) -> Result<(), ResidentError> {
        let deferred_models = generation._operation.seal_model_release_deferrals();
        let generation_id = generation.id;
        let timeout = generation.timeout;
        let mut worker = match self.resident_worker() {
            Ok(worker) => worker,
            Err(error) => {
                // Never disarm a live generation watchdog merely because the
                // parent can no longer reach the worker lock/protocol.
                generation.deadline.expire_now();
                return Err(error);
            }
        };
        let result = worker.under_deadline(
            &generation.deadline,
            timeout,
            DeadlineKind::Execution,
            |worker, deadline| {
                worker.request(
                    &Request::ReleaseGeneration {
                        generation: generation_id,
                    },
                    deadline,
                )
            },
        );
        let mut teardown = match result {
            Ok(Response::Resident(Ok(ResidentResponse::GenerationReleased(id))))
                if id == generation_id =>
            {
                Ok(())
            }
            Ok(Response::Resident(Err(error))) => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::Teardown(error))
            }
            Ok(_) => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::UnexpectedResponse)
            }
            Err(error) => {
                worker.invalidate_after_protocol_failure();
                Err(map_resident_teardown_worker_error(error))
            }
        };
        if teardown.is_ok() {
            for release in deferred_models {
                let released = worker.under_deadline(
                    &generation.deadline,
                    timeout,
                    DeadlineKind::Execution,
                    |worker, deadline| worker.request(&release.request(), deadline),
                );
                teardown = match released {
                    Ok(ref response) if release.acknowledged(response) => Ok(()),
                    Ok(Response::Resident(Err(error))) => {
                        worker.invalidate_after_protocol_failure();
                        Err(ResidentError::Teardown(error))
                    }
                    Ok(_) => {
                        worker.invalidate_after_protocol_failure();
                        Err(ResidentError::UnexpectedResponse)
                    }
                    Err(error) => {
                        worker.invalidate_after_protocol_failure();
                        Err(map_resident_teardown_worker_error(error))
                    }
                };
                if teardown.is_err() {
                    break;
                }
            }
        }
        worker
            .finish_deadline(generation.deadline, timeout, DeadlineKind::Execution)
            .map_err(map_resident_worker_error)?;
        teardown
    }

    pub(crate) fn release_resident_model(&self, model: ResidentModel) -> Result<(), ResidentError> {
        self.release_object(DeferredRelease::Model(model.id))
    }

    fn release_object(&self, release: DeferredRelease) -> Result<(), ResidentError> {
        let Some(_operation) = self
            .operation_gate
            .acquire_or_defer_release(release)
            .map_err(|()| map_resident_worker_error(WorkerError::ReentrantOperation))?
        else {
            return Ok(());
        };
        let mut worker = worker_for_model_teardown(self.worker.lock())?;
        if let Some(termination) = worker.termination() {
            return Err(ResidentError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        let response = match worker.with_deadline(
            self.timeouts.execution_timeout(),
            DeadlineKind::Execution,
            |worker, deadline| worker.request(&release.request(), deadline),
        ) {
            Ok(response) => response,
            Err(error) => {
                // Teardown is stricter than an ordinary request: even a
                // provably local preparation failure would leave the model
                // resident, so no other handle may continue on this worker.
                worker.invalidate_after_protocol_failure();
                return Err(map_resident_teardown_worker_error(error));
            }
        };
        match response {
            ref response if release.acknowledged(response) => Ok(()),
            Response::Resident(Err(error)) => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::Teardown(error))
            }
            _ => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::UnexpectedResponse)
            }
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

    /// Run a source-level program with dynamically sized inputs and outputs.
    fn exec_values<'a>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(SafeExecError::UnknownArtifact);
        }
        let _operation = self.begin_operation().map_err(map_exec_worker_error)?;
        let ipc = self.parent_ipc()?;
        let (buffers, wire_args) = Self::encode_parent_arguments(ipc, &args)?;
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

        let response = worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    worker.request(
                        &Request::Execute {
                            artifact: artifact.index(),
                            name: name.to_string(),
                            buffers,
                            args: wire_args,
                        },
                        deadline,
                    )
                },
            )
            .map_err(map_exec_worker_error)?;
        let execution = execution_from_response(&mut worker, response)?;
        let values = Self::decode_child_outputs(ipc, execution);
        let released = worker.with_deadline(
            self.timeouts.execution_timeout(),
            DeadlineKind::Execution,
            |worker, deadline| worker.request(&Request::ReleaseOutputs, deadline),
        );
        match released {
            Ok(Response::OutputsReleased) => match values {
                Err(error @ (SafeExecError::Transport(_) | SafeExecError::UnexpectedResponse)) => {
                    Err(worker.reject_protocol_response(error))
                }
                result => result,
            },
            Ok(_) => {
                worker.invalidate_after_protocol_failure();
                Err(SafeExecError::UnexpectedResponse)
            }
            Err(error) => {
                worker.invalidate_after_protocol_failure();
                Err(map_exec_worker_error(error))
            }
        }
    }

    /// Exports parent-owned arguments as views for the child to copy into its own allocations.
    fn parent_ipc(&self) -> Result<&IpcTransport, SafeExecError> {
        self.parent_ipc.as_ref().ok_or_else(|| {
            SafeExecError::InvalidRequest(
                "raw Value execution is unavailable on a resident-protocol runtime".to_string(),
            )
        })
    }

    fn encode_parent_arguments(
        ipc: &IpcTransport,
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
                    if memory.dialect() != ipc.dialect() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = ipc.export_view(memory.as_ref())?;
                    let buffer_index = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                    WireValue::MemOwn {
                        buffer: buffer_index,
                        view_offset: exported.view_offset(),
                        byte_len: exported.byte_len(),
                    }
                }
                Value::MemRef(memory) => {
                    if memory.dialect() != ipc.dialect() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = ipc.export_view(*memory)?;
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
            ipc.synchronize()?;
        }
        Ok((buffers, values))
    }

    /// Copies child-owned outputs into parent-owned allocations before releasing them remotely.
    fn decode_child_outputs(
        ipc: &IpcTransport,
        execution: WireExecution,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        let imported = import_ipc_buffers(ipc, execution.buffers).map_err(|error| {
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

fn worker_for_model_teardown(
    worker: std::sync::LockResult<std::sync::MutexGuard<'_, WorkerProcess>>,
) -> Result<std::sync::MutexGuard<'_, WorkerProcess>, ResidentError> {
    match worker {
        Ok(worker) => Ok(worker),
        Err(poisoned) => {
            // A panic while the worker lock was held leaves request progress
            // unknowable. Recover ownership solely to terminate the process;
            // silently abandoning a live resident model here would defeat the
            // teardown acknowledgement protocol.
            let mut worker = poisoned.into_inner();
            worker.invalidate_after_protocol_failure();
            Err(ResidentError::Transport(
                "worker lock was poisoned during model teardown".to_string(),
            ))
        }
    }
}

fn execution_from_response(
    worker: &mut WorkerProcess,
    response: Response,
) -> Result<WireExecution, SafeExecError> {
    match response {
        Response::Executed(Ok(execution)) => Ok(execution),
        Response::Executed(Err(RemoteExecError::UnknownArtifact)) => {
            Err(SafeExecError::UnknownArtifact)
        }
        Response::Executed(Err(RemoteExecError::Runtime(error))) => {
            Err(SafeExecError::Runtime(error))
        }
        Response::Executed(Err(RemoteExecError::Memory(error))) => {
            Err(SafeExecError::RemoteMemory(error))
        }
        Response::Executed(Err(RemoteExecError::Protocol(error))) => Err(worker
            .reject_protocol_response(SafeExecError::Transport(format!(
                "child execution protocol failed: {error}"
            )))),
        _ => Err(worker.reject_protocol_response(SafeExecError::UnexpectedResponse)),
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
/// Call it before starting threads or child processes: in worker mode fd 3 is
/// the inherited asset socket. The entrypoint duplicates that descriptor for
/// owned use and marks the original close-on-exec without closing it, preventing
/// compiler inheritance without claiming a descriptor from a spoofed caller.
pub fn run_safe_runtime_child_if_requested() -> Result<bool, ChildMainError> {
    if env::var_os(CHILD_MODE_ENV).is_none() {
        return Ok(false);
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    // Worker mode is selected through the environment, which is not itself
    // proof that fd 3 is valid or uniquely owned. Duplicate it first: the
    // syscall rejects an invalid descriptor, and only the returned descriptor
    // is transferred into the owned UnixStream below. The bootstrap descriptor
    // is marked close-on-exec, not closed, so this safe entrypoint neither
    // claims nor double-closes a descriptor owned by a spoofed caller while a
    // genuine worker's compiler descendants cannot retain the socket endpoint.
    let asset_socket = duplicate_unix_stream(CHILD_ASSET_SOCKET_FD).map_err(|error| {
        ChildMainError::Protocol(format!("invalid worker asset socket: {error}"))
    })?;
    run_child_loop(stdin.lock(), stdout.lock(), &asset_socket)?;
    Ok(true)
}

fn duplicate_unix_stream(descriptor: RawFd) -> io::Result<UnixStream> {
    let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 0) };
    if duplicate == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `F_DUPFD_CLOEXEC` returned a fresh, valid descriptor that this
    // function now exclusively owns.
    let duplicate = unsafe { OwnedFd::from_raw_fd(duplicate) };
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags == -1
        || unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == -1
    {
        return Err(io::Error::last_os_error());
    }
    Ok(UnixStream::from(duplicate))
}

fn run_child_loop(
    mut reader: impl Read,
    mut writer: impl io::Write,
    asset_socket: &UnixStream,
) -> Result<(), ChildMainError> {
    let request = read_request(&mut reader)?.ok_or(ChildMainError::ExpectedInitialization)?;
    let (backend, asset_owner) = match request {
        Request::Initialize { backend } => (backend, false),
        Request::InitializeAssets { backend } => (backend, true),
        _ => return Err(ChildMainError::ExpectedInitialization),
    };

    let runtime = match Runtime::with_backend(backend) {
        Ok(runtime) => runtime,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    let dialect = runtime.dialect();
    let ipc = IpcTransport::from_runtime(&runtime);
    let mut assets = AssetStore::new(dialect);
    let mut resident = ResidentStore::new();
    let mut artifacts = HashMap::<usize, Arc<RuntimeArtifact>>::new();
    let mut next_artifact = 0_usize;
    write_response(&mut writer, &Response::Initialized(Ok(dialect)))?;

    let mut pending_outputs = Vec::new();
    while let Some(request) = read_request(&mut reader)? {
        // The owner never loads supplied programs. Its process and allocations
        // survive faults in the separate execution workers.
        if asset_owner
            && !matches!(
                request,
                Request::AttachAsset { .. }
                    | Request::ExportAsset { .. }
                    | Request::AssetUsage
                    | Request::Shutdown
            )
        {
            return Err(ChildMainError::Protocol(
                "asset owner received an execution request".into(),
            ));
        }
        match request {
            Request::Initialize { .. } | Request::InitializeAssets { .. } => {
                return Err(ChildMainError::AlreadyInitialized);
            }
            Request::LoadSources { sources } => {
                let result = runtime
                    .load_sources(sources.iter().map(String::as_str))
                    .map(|artifact| {
                        let id = next_artifact;
                        next_artifact = next_artifact
                            .checked_add(1)
                            .expect("artifact ID space exhausted");
                        let entries = artifact.entry_points().to_vec();
                        artifacts.insert(id, Arc::new(artifact));
                        (id, entries)
                    })
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Loaded(result))?;
            }
            Request::AttachAsset { key, byte_len } => {
                // The descriptor channel is part of request framing. A failed
                // receive may have consumed an unknown amount of ancillary
                // data, so it is a fatal protocol error rather than a reusable
                // structured asset rejection.
                let file = receive_asset_file(asset_socket)?;
                let result = assets
                    .attach(key, byte_len, file)
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::AssetUsage => {
                let (count, uploaded, device) = assets.usage();
                write_response(&mut writer, &Response::AssetUsage(count, uploaded, device))?;
            }
            Request::ExportAsset { asset } => {
                if !asset_owner {
                    return Err(ChildMainError::Protocol(
                        "only the asset owner may export allocations".into(),
                    ));
                }
                match assets.export(asset) {
                    Ok((file, byte_len, allocation_len)) => {
                        write_response(
                            &mut writer,
                            &Response::Exported(Ok((byte_len, allocation_len))),
                        )?;
                        send_file(asset_socket, &file)
                            .map_err(|error| ChildMainError::Protocol(error.to_string()))?;
                    }
                    Err(error) => {
                        write_response(&mut writer, &Response::Exported(Err(error.to_string())))?
                    }
                }
            }
            Request::ImportAsset {
                asset,
                byte_len,
                allocation_len,
            } => {
                let file = receive_asset_file(asset_socket)?;
                let result = assets
                    .import(asset, byte_len, allocation_len, file)
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::BindModel { binding } => {
                let result = artifacts
                    .get(&binding.artifact)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown artifact {}", binding.artifact))
                    .and_then(|artifact| resident.bind_model(artifact, &assets, binding))
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
                write_response(
                    &mut writer,
                    &Response::Resident(Ok(ResidentResponse::GenerationReleased(generation))),
                )?;
            }
            Request::ReleaseArtifact { artifact } => {
                artifacts.remove(&artifact);
                write_response(
                    &mut writer,
                    &Response::Resident(Ok(ResidentResponse::ArtifactReleased(artifact))),
                )?;
            }
            Request::ReleaseModel { model } => {
                resident.release_model(model);
                write_response(
                    &mut writer,
                    &Response::Resident(Ok(ResidentResponse::ModelReleased(model))),
                )?;
            }
            Request::Shutdown => return Ok(()),
            Request::ReleaseOutputs => {
                pending_outputs.clear();
                write_response(&mut writer, &Response::OutputsReleased)?;
            }
            Request::Execute {
                artifact,
                name,
                buffers,
                args,
            } => {
                let response = if pending_outputs.is_empty() {
                    match artifacts.get(&artifact) {
                        Some(artifact) => execute_in_child(
                            &ipc,
                            artifact,
                            &name,
                            buffers,
                            args,
                            &mut pending_outputs,
                        )?,
                        None => Response::Executed(Err(RemoteExecError::UnknownArtifact)),
                    }
                } else {
                    Response::Executed(Err(RemoteExecError::Protocol(
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
    ipc: &IpcTransport,
    artifact: &RuntimeArtifact,
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

    let values = match artifact.exec_values(name, args) {
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

fn receive_asset_file(asset_socket: &UnixStream) -> Result<File, ChildMainError> {
    receive_file(asset_socket).map_err(|error| {
        ChildMainError::Protocol(format!("asset descriptor transport failed: {error}"))
    })
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

#[derive(Debug, Default)]
struct StderrTail {
    bytes: Vec<u8>,
    truncated: bool,
}

impl StderrTail {
    fn push(&mut self, bytes: &[u8]) {
        let retained = bytes.len().min(STDERR_TAIL_BYTES);
        let required = self.bytes.len().saturating_add(retained);
        let discard = required.saturating_sub(STDERR_TAIL_BYTES);
        if bytes.len() != retained || discard != 0 {
            self.truncated = true;
        }
        if discard != 0 {
            self.bytes.drain(..discard);
        }
        self.bytes
            .extend_from_slice(&bytes[bytes.len() - retained..]);
    }

    fn render(&self) -> String {
        let tail = String::from_utf8_lossy(&self.bytes);
        if !self.truncated {
            tail.trim().to_string()
        } else {
            format!(
                "[... stderr truncated; retaining final {} bytes ...]\n{}",
                self.bytes.len(),
                tail.trim()
            )
        }
    }
}

/// A nonblocking protocol reader that waits only until the operation's one
/// absolute deadline. Descendants retaining a killed worker's pipe therefore
/// cannot strand the caller in `read(2)` after the watchdog fires.
struct DeadlineReader<'a, R> {
    inner: &'a mut R,
    descriptor: RawFd,
    deadline: &'a ProcessGroupDeadline,
}

impl<'a, R> DeadlineReader<'a, R> {
    fn new(inner: &'a mut R, descriptor: RawFd, deadline: &'a ProcessGroupDeadline) -> Self {
        Self {
            inner,
            descriptor,
            deadline,
        }
    }
}

impl<R: Read> Read for DeadlineReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.read(buffer) {
                Ok(read) => return Ok(read),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLIN)?;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

/// The write-side counterpart to [`DeadlineReader`]. `write_frame` may make
/// partial progress on a full pipe, but every retry consumes the same deadline.
struct DeadlineWriter<'a, W> {
    inner: &'a mut W,
    descriptor: RawFd,
    deadline: &'a ProcessGroupDeadline,
}

impl<'a, W> DeadlineWriter<'a, W> {
    fn new(inner: &'a mut W, descriptor: RawFd, deadline: &'a ProcessGroupDeadline) -> Self {
        Self {
            inner,
            descriptor,
            deadline,
        }
    }
}

impl<W: Write> Write for DeadlineWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.write(buffer) {
                Ok(written) => return Ok(written),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLOUT)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.flush() {
                Ok(()) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLOUT)?;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

fn set_nonblocking(descriptor: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn ensure_waitable_children() -> Result<(), SafeInitError> {
    let mut disposition = std::mem::MaybeUninit::<libc::sigaction>::uninit();
    if unsafe { libc::sigaction(libc::SIGCHLD, std::ptr::null(), disposition.as_mut_ptr()) } == -1 {
        return Err(SafeInitError::ChildSignal(io::Error::last_os_error()));
    }
    let disposition = unsafe { disposition.assume_init() };
    if !sigchld_is_waitable(disposition.sa_sigaction, disposition.sa_flags) {
        Err(SafeInitError::UnwaitableChildren)
    } else {
        Ok(())
    }
}

fn sigchld_is_waitable(handler: libc::sighandler_t, flags: libc::c_int) -> bool {
    handler != libc::SIG_IGN && flags & libc::SA_NOCLDWAIT == 0
}

/// Owns the usable lifetime of one private process-group ID.
///
/// Linux may reuse a numeric PID/PGID as soon as the group leader is reaped.
/// Every watchdog signal and every kill-plus-`try_wait` therefore takes the
/// same gate; the successful reap retires the ID before releasing that gate. A
/// stale generation watchdog can observe retirement, but can never signal a
/// later process that happened to receive the same numeric ID. Reaping is
/// deliberately exposed only together with a preceding group-wide kill, so a
/// descendant cannot outlive the leader whose PID pins this identity.
#[derive(Debug)]
struct ManagedProcessGroup {
    id: u32,
    state: Mutex<ManagedProcessGroupState>,
    lifecycle_gate: Arc<WorkerLifecycleGate>,
    retirement: std::sync::OnceLock<WorkerRetirementPermit>,
}

#[derive(Debug)]
struct ManagedProcessGroupState {
    active: bool,
    kill_count: usize,
}

impl ManagedProcessGroup {
    fn new(id: u32, lifecycle_gate: Arc<WorkerLifecycleGate>) -> Self {
        Self {
            id,
            state: Mutex::new(ManagedProcessGroupState {
                active: true,
                kill_count: 0,
            }),
            lifecycle_gate,
            retirement: std::sync::OnceLock::new(),
        }
    }

    #[cfg(test)]
    fn new_test(id: u32) -> Self {
        Self::new(id, Arc::new(WorkerLifecycleGate::default()))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, ManagedProcessGroupState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn kill(&self) {
        let mut state = self.state();
        if !state.active {
            return;
        }
        self.begin_retirement_while_active();
        state.kill_count = state.kill_count.saturating_add(1);
        kill_process_group(self.id);
    }

    /// Close process-wide worker admission before this live leader enters any
    /// terminal path. The OnceLock makes concurrent watchdog, protocol, and
    /// Drop transitions idempotent without a poisonable lifecycle mutex.
    fn begin_retirement(&self) {
        let state = self.state();
        if state.active {
            self.begin_retirement_while_active();
        }
    }

    fn begin_retirement_while_active(&self) {
        self.retirement
            .get_or_init(|| self.lifecycle_gate.begin_retirement());
    }

    fn finish_retirement_after_wait(&self) {
        if let Some(retirement) = self.retirement.get() {
            retirement.finish_after_wait();
        }
    }

    /// Kill every current member while the numeric PGID is still protected,
    /// then reap the leader under that same gate. Keeping these inseparable is
    /// what makes both descendant cleanup and PGID retirement hard to bypass.
    fn kill_and_try_wait(&self, child: &mut Child) -> io::Result<Option<ExitStatus>> {
        let mut state = self.state();
        if state.active {
            self.begin_retirement_while_active();
            state.kill_count = state.kill_count.saturating_add(1);
            kill_process_group(self.id);
        }
        let status = child.try_wait()?;
        if status.is_some() {
            state.active = false;
            self.finish_retirement_after_wait();
        }
        Ok(status)
    }

    /// Used only on an already-killed exceptional path before a blocking
    /// background wait. No future signal may use this numeric identity.
    fn retire(&self) {
        self.state().active = false;
    }

    #[cfg(test)]
    fn kill_count(&self) -> usize {
        self.state().kill_count
    }
}

#[derive(Debug)]
struct ReapResources {
    child: Option<Child>,
    process_group: Arc<ManagedProcessGroup>,
    stderr_reader: Option<JoinHandle<()>>,
    scratch: Option<tempfile::TempDir>,
    #[cfg(test)]
    test_barrier: Option<Box<LifecycleTestBarrier>>,
}

#[cfg(test)]
#[derive(Debug)]
struct LifecycleTestBarrier {
    entered: mpsc::SyncSender<()>,
    release: mpsc::Receiver<()>,
}

impl ReapResources {
    fn new(
        child: Option<Child>,
        process_group: Arc<ManagedProcessGroup>,
        stderr_reader: Option<JoinHandle<()>>,
        scratch: Option<tempfile::TempDir>,
    ) -> Self {
        Self {
            child,
            process_group,
            stderr_reader,
            scratch,
            #[cfg(test)]
            test_barrier: None,
        }
    }
}

/// A reaper is created fallibly before its worker child. Teardown can then
/// transfer ownership without attempting to create a thread from `Drop`, when
/// failure would otherwise strand a killed child without a waiter.
#[derive(Debug)]
struct BackgroundReaper {
    sender: Option<mpsc::SyncSender<ReapResources>>,
    handle: Option<JoinHandle<()>>,
}

impl BackgroundReaper {
    fn start() -> io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle = thread::Builder::new()
            .name("catena-safe-runtime-reaper".to_string())
            .spawn(move || {
                if let Ok(resources) = receiver.recv() {
                    reap_resources(resources);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn handoff(&mut self, resources: ReapResources) -> Result<(), ReapResources> {
        let Some(sender) = self.sender.take() else {
            return Err(resources);
        };
        match sender.send(resources) {
            Ok(()) => {
                // Detach only after the live receiver has accepted ownership.
                self.handle.take();
                Ok(())
            }
            Err(mpsc::SendError(resources)) => Err(resources),
        }
    }
}

impl Drop for BackgroundReaper {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn reap_resources(mut resources: ReapResources) {
    #[cfg(test)]
    if resources.child.is_some()
        && let Some(barrier) = resources.test_barrier.take()
    {
        let _ = barrier.entered.send(());
        let _ = barrier.release.recv();
    }
    if let Some(mut child) = resources.child.take() {
        resources.process_group.kill();
        loop {
            match resources.process_group.kill_and_try_wait(&mut child) {
                Ok(Some(_)) => break,
                Ok(None) => thread::sleep(Duration::from_millis(5)),
                Err(_) => {
                    // The group has already received SIGKILL. Retire its
                    // numeric identity before falling back to a blocking wait,
                    // so stale watchdogs cannot target a reused PGID.
                    resources.process_group.retire();
                    if child.wait().is_ok() {
                        resources.process_group.finish_retirement_after_wait();
                    }
                    break;
                }
            }
        }
    }
    if let Some(stderr_reader) = resources.stderr_reader {
        let _ = stderr_reader.join();
    }
    // TempDir cleanup can traverse compiler output. Keep it on the lifecycle
    // thread so a forced operation and WorkerProcess::drop remain bounded.
    drop(resources.scratch.take());
}

#[derive(Debug)]
struct WorkerProcess {
    child: Option<Child>,
    process_group: Arc<ManagedProcessGroup>,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    asset_socket: UnixStream,
    stderr_tail: Arc<Mutex<StderrTail>>,
    stderr_done: mpsc::Receiver<()>,
    stderr_reader: Option<JoinHandle<()>>,
    background_reaper: Option<BackgroundReaper>,
    scratch: Option<tempfile::TempDir>,
    termination: Option<Termination>,
    imported_assets: HashSet<u64>,
    #[cfg(test)]
    force_background_handoff: bool,
    #[cfg(test)]
    reaper_test_barrier: Option<Box<LifecycleTestBarrier>>,
    #[cfg(test)]
    drop_retirement_test_barrier: Option<Box<LifecycleTestBarrier>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeadlineKind {
    Initialize,
    Compile,
    Execution,
}

#[derive(Debug, Error)]
enum WorkerError {
    #[error("operation re-entered the same SafeRuntime session from its active callback")]
    ReentrantOperation,
    #[error("local request preparation failed: {0}")]
    LocalRequest(#[source] FrameEncodeError),
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error(transparent)]
    AssetDescriptor(#[from] FdTransportError),
    #[error("failed to wait for SafeRuntime child: {0}")]
    Wait(#[source] io::Error),
    #[error("failed to start the SafeRuntime deadline watchdog: {0}")]
    Watchdog(#[source] io::Error),
    #[error("SafeRuntime child terminated")]
    Terminated(Termination),
    #[error("SafeRuntime worker is invalid after an earlier failed operation")]
    Invalidated,
    #[error("SafeRuntime operation exceeded {timeout:?}")]
    TimedOut {
        kind: DeadlineKind,
        timeout: Duration,
    },
}

impl WorkerError {
    fn invalidates_worker(&self) -> bool {
        !matches!(self, Self::ReentrantOperation | Self::LocalRequest(_))
    }
}

impl WorkerProcess {
    fn spawn(executable: &Path) -> Result<Self, SafeInitError> {
        let mut command = Command::new(executable);
        command.env(CHILD_MODE_ENV, "1");
        Self::spawn_command_with_gate(command, executable, worker_lifecycle_gate())
    }

    #[cfg(test)]
    fn spawn_command(command: Command, executable: &Path) -> Result<Self, SafeInitError> {
        Self::spawn_command_with_gate(
            command,
            executable,
            Arc::new(WorkerLifecycleGate::default()),
        )
    }

    fn spawn_command_with_gate(
        command: Command,
        executable: &Path,
        lifecycle_gate: Arc<WorkerLifecycleGate>,
    ) -> Result<Self, SafeInitError> {
        let scratch_parent = env::temp_dir();
        Self::spawn_command_in_with_gate(command, executable, &scratch_parent, lifecycle_gate)
    }

    #[cfg(test)]
    fn spawn_command_in(
        command: Command,
        executable: &Path,
        scratch_parent: &Path,
    ) -> Result<Self, SafeInitError> {
        Self::spawn_command_in_with_gate(
            command,
            executable,
            scratch_parent,
            Arc::new(WorkerLifecycleGate::default()),
        )
    }

    fn spawn_command_in_with_gate(
        mut command: Command,
        executable: &Path,
        scratch_parent: &Path,
        lifecycle_gate: Arc<WorkerLifecycleGate>,
    ) -> Result<Self, SafeInitError> {
        // This permit linearizes worker admission against every detached
        // retirement in the process. It is released once construction either
        // owns a complete WorkerProcess or has registered any spawned child
        // with its reaper.
        let _spawn_permit = lifecycle_gate.begin_spawn()?;
        let scratch = tempfile::Builder::new()
            .prefix("catena-safe-runtime-")
            .tempdir_in(scratch_parent)
            .map_err(|source| SafeInitError::Scratch {
                parent: scratch_parent.to_path_buf(),
                source,
            })?;
        let scratch_path =
            fs::canonicalize(scratch.path()).map_err(|source| SafeInitError::Scratch {
                parent: scratch_parent.to_path_buf(),
                source,
            })?;
        let (asset_socket, child_asset_socket) =
            UnixStream::pair().map_err(SafeInitError::AssetTransport)?;
        let child_asset_fd = child_asset_socket.as_raw_fd();
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Every tempfile made by the worker or its compiler descendants is
            // scoped beneath a parent-owned directory. The parent retains that
            // directory until the private process group has been killed and its
            // leader reaped, so SIGKILL cannot strand Catena build scratch.
            .env("TMPDIR", scratch_path);
        // The worker and every compiler it spawns live in a private process
        // group. A compile deadline can therefore stop hipcc/clang descendants
        // as well as the protocol child that is blocked waiting for them.
        command.process_group(0);
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
        // Create the teardown owner before there is a child it might need to
        // own. Thread exhaustion is therefore a construction error, never a
        // reason for `Drop` to lose the only wait handle.
        let mut background_reaper =
            BackgroundReaper::start().map_err(SafeInitError::LifecycleThread)?;
        let mut child = command.spawn().map_err(|source| SafeInitError::Spawn {
            executable: executable.to_path_buf(),
            source,
        })?;
        let process_group = Arc::new(ManagedProcessGroup::new(child.id(), lifecycle_gate.clone()));
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
        let stderr_tail = Arc::new(Mutex::new(StderrTail::default()));
        let captured_tail = stderr_tail.clone();
        let (stderr_finished, stderr_done) = mpsc::channel();
        if let Err(error) =
            set_nonblocking(stdin.as_raw_fd()).and_then(|()| set_nonblocking(stdout.as_raw_fd()))
        {
            // A killed process can still be stuck in uninterruptible sleep.
            // Give ownership to a reaper rather than making construction an
            // unbounded wait on this exceptional setup path.
            process_group.kill();
            drop(stderr);
            let resources = ReapResources::new(Some(child), process_group, None, Some(scratch));
            if let Err(resources) = background_reaper.handoff(resources) {
                // The pre-created receiver has no fallible work before recv,
                // so this is only a defensive ownership-preserving fallback.
                reap_resources(resources);
            }
            return Err(SafeInitError::Transport(format!(
                "failed to make SafeRuntime protocol pipes nonblocking: {error}"
            )));
        }
        let stderr_reader = thread::Builder::new()
            .name("catena-safe-runtime-stderr".to_string())
            .spawn(move || {
                let mut bytes = [0_u8; 8 * 1024];
                loop {
                    match stderr.read(&mut bytes) {
                        Ok(0) => break,
                        Ok(read) => {
                            let Ok(mut tail) = captured_tail.lock() else {
                                break;
                            };
                            tail.push(&bytes[..read]);
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
                let _ = stderr_finished.send(());
            });
        let stderr_reader = match stderr_reader {
            Ok(stderr_reader) => stderr_reader,
            Err(error) => {
                process_group.kill();
                let resources = ReapResources::new(Some(child), process_group, None, Some(scratch));
                if let Err(resources) = background_reaper.handoff(resources) {
                    reap_resources(resources);
                }
                return Err(SafeInitError::LifecycleThread(error));
            }
        };

        Ok(Self {
            child: Some(child),
            process_group,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            asset_socket,
            stderr_tail,
            stderr_done,
            stderr_reader: Some(stderr_reader),
            background_reaper: Some(background_reaper),
            scratch: Some(scratch),
            termination: None,
            imported_assets: HashSet::new(),
            #[cfg(test)]
            force_background_handoff: false,
            #[cfg(test)]
            reaper_test_barrier: None,
            #[cfg(test)]
            drop_retirement_test_barrier: None,
        })
    }

    fn send(
        &mut self,
        request: &Request,
        deadline: &ProcessGroupDeadline,
    ) -> Result<(), WorkerError> {
        self.ensure_available()?;
        // Serialization and the size check complete before touching the
        // protocol pipe. Their failures are therefore local request errors,
        // unlike every subsequent I/O error whose partial progress is unknown.
        let frame = encode_frame(request).map_err(WorkerError::LocalRequest)?;
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(WorkerError::Invalidated);
        };
        let descriptor = stdin.as_raw_fd();
        write_encoded_frame(
            &mut DeadlineWriter::new(stdin, descriptor, deadline),
            &frame,
        )
        .map_err(WorkerError::Protocol)
    }

    fn receive(&mut self, deadline: &ProcessGroupDeadline) -> Result<Response, WorkerError> {
        let descriptor = self.stdout.get_ref().as_raw_fd();
        let response = read_frame(&mut DeadlineReader::new(
            &mut self.stdout,
            descriptor,
            deadline,
        ))
        .map_err(WorkerError::Protocol)?;
        match response {
            Some(response) => Ok(response),
            None => Err(WorkerError::Terminated(self.reap(deadline)?)),
        }
    }

    fn request(
        &mut self,
        request: &Request,
        deadline: &ProcessGroupDeadline,
    ) -> Result<Response, WorkerError> {
        self.send(request, deadline)?;
        self.receive(deadline)
    }

    fn with_deadline<T>(
        &mut self,
        timeout: Duration,
        kind: DeadlineKind,
        operation: impl FnOnce(&mut Self, &ProcessGroupDeadline) -> Result<T, WorkerError>,
    ) -> Result<T, WorkerError> {
        let deadline = self.arm_deadline(timeout)?;
        let result = operation(self, &deadline);
        self.finish_deadline(deadline, timeout, kind)?;
        if result.as_ref().is_err_and(WorkerError::invalidates_worker) {
            // Any failed protocol/descriptor transaction may have made partial
            // progress. Keep the safe API fail-closed: callers need not remember
            // to inspect an error classifier before attempting another request.
            // Request encoding and size validation are the sole exception:
            // both finish before the protocol pipe is touched.
            self.invalidate_after_protocol_failure();
        }
        result
    }

    fn arm_deadline(&mut self, timeout: Duration) -> Result<ProcessGroupDeadline, WorkerError> {
        self.ensure_available()?;
        match ProcessGroupDeadline::arm(self.process_group.clone(), timeout) {
            Ok(deadline) => Ok(deadline),
            Err(error) => {
                // Running without the promised watchdog is not a recoverable
                // downgrade. Kill and hand off while the worker is still owned.
                self.invalidate_after_protocol_failure();
                Err(WorkerError::Watchdog(error))
            }
        }
    }

    fn under_deadline<T>(
        &mut self,
        deadline: &ProcessGroupDeadline,
        timeout: Duration,
        kind: DeadlineKind,
        operation: impl FnOnce(&mut Self, &ProcessGroupDeadline) -> Result<T, WorkerError>,
    ) -> Result<T, WorkerError> {
        if deadline.timed_out() {
            self.invalidate_after_timeout();
            return Err(WorkerError::TimedOut { kind, timeout });
        }
        let result = operation(self, deadline);
        if deadline.timed_out() {
            self.invalidate_after_timeout();
            Err(WorkerError::TimedOut { kind, timeout })
        } else {
            if result.as_ref().is_err_and(WorkerError::invalidates_worker) {
                self.invalidate_after_protocol_failure();
            }
            result
        }
    }

    fn finish_deadline(
        &mut self,
        deadline: ProcessGroupDeadline,
        timeout: Duration,
        kind: DeadlineKind,
    ) -> Result<(), WorkerError> {
        if deadline.finish() {
            self.invalidate_after_timeout();
            Err(WorkerError::TimedOut { kind, timeout })
        } else {
            Ok(())
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

    fn ensure_available(&self) -> Result<(), WorkerError> {
        if let Some(termination) = &self.termination {
            Err(WorkerError::Terminated(termination.clone()))
        } else if self.stdin.is_none() || self.child.is_none() {
            Err(WorkerError::Invalidated)
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    fn child_id(&self) -> Result<u32, WorkerError> {
        self.child
            .as_ref()
            .map(Child::id)
            .ok_or(WorkerError::Invalidated)
    }

    fn reap(&mut self, deadline: &ProcessGroupDeadline) -> Result<Termination, WorkerError> {
        if let Some(termination) = &self.termination {
            return Ok(termination.clone());
        }
        // EOF is never a successful response. Kill the whole private group
        // before reaping its leader so a compiler or other descendant cannot
        // survive the worker that owned it.
        self.process_group.kill();
        self.stdin.take();
        let Some(initial_wait) = deadline
            .remaining()
            .map(|remaining| remaining.min(KILL_REAP_TIMEOUT))
        else {
            self.invalidate_after_timeout();
            return Err(WorkerError::Invalidated);
        };
        let status = match self.wait_for_exit(initial_wait)? {
            Some(status) => status,
            None => {
                self.process_group.kill();
                let Some(kill_wait) = deadline
                    .remaining()
                    .map(|remaining| remaining.min(KILL_REAP_TIMEOUT))
                else {
                    self.invalidate_after_timeout();
                    return Err(WorkerError::Invalidated);
                };
                let Some(status) = self.wait_for_exit(kill_wait)? else {
                    let timed_out = deadline.remaining().is_none();
                    self.handoff_background_reaper();
                    return Err(if timed_out {
                        WorkerError::Invalidated
                    } else {
                        WorkerError::Wait(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "SafeRuntime child could not be reaped",
                        ))
                    });
                };
                status
            }
        };
        let stderr_wait = deadline
            .remaining()
            .unwrap_or_default()
            .min(Duration::from_millis(100));
        Ok(self.record_exit_with_stderr_wait(status, stderr_wait))
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, WorkerError> {
        // This helper is terminal-only: each waitpid probe first sends the
        // group-wide kill under the lifecycle gate. It must never be used to
        // observe a healthy worker.
        let deadline = Instant::now() + timeout;
        loop {
            #[cfg(test)]
            if self.force_background_handoff {
                self.process_group.kill();
                return Ok(None);
            }
            let child = self.child.as_mut().ok_or(WorkerError::Invalidated)?;
            if let Some(status) = self
                .process_group
                .kill_and_try_wait(child)
                .map_err(WorkerError::Wait)?
            {
                return Ok(Some(status));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            thread::sleep((deadline - now).min(Duration::from_millis(5)));
        }
    }

    fn invalidate_after_timeout(&mut self) {
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        // The watchdog has already killed this private process group. Repeat
        // the signal defensively, but never extend the operation past its
        // absolute deadline to wait for an uninterruptible child. A dedicated
        // background reaper owns that pathological process until `wait(2)` and
        // stderr capture can finish.
        self.process_group.kill();
        let status = self.try_reap_immediately();
        if let Some(status) = status {
            self.record_exit_now(status);
        } else {
            self.handoff_background_reaper();
        }
    }

    fn invalidate_after_protocol_failure(&mut self) {
        if self.termination.is_some() || self.child.is_none() {
            return;
        }
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        self.process_group.kill();
        let status = self.try_reap_immediately();
        if let Some(status) = status {
            self.record_exit_now(status);
        } else {
            self.handoff_background_reaper();
        }
    }

    /// Convert a semantically impossible response into a fail-closed error.
    /// Framing succeeded in this case, but request/response synchronization can
    /// no longer be established, so retaining the worker would be unsafe.
    fn reject_protocol_response<E>(&mut self, error: E) -> E {
        self.invalidate_after_protocol_failure();
        error
    }

    fn try_reap_immediately(&mut self) -> Option<ExitStatus> {
        #[cfg(test)]
        if self.force_background_handoff {
            self.process_group.kill();
            return None;
        }
        self.child
            .as_mut()
            .and_then(|child| self.process_group.kill_and_try_wait(child).ok().flatten())
    }

    fn record_exit(&mut self, status: ExitStatus) -> Termination {
        self.record_exit_with_stderr_wait(status, Duration::from_millis(100))
    }

    fn record_exit_with_stderr_wait(
        &mut self,
        status: ExitStatus,
        stderr_wait: Duration,
    ) -> Termination {
        if self.stderr_done.recv_timeout(stderr_wait).is_ok()
            && let Some(stderr_reader) = self.stderr_reader.take()
        {
            let _ = stderr_reader.join();
        }
        self.record_exit_now(status)
    }

    fn record_exit_now(&mut self, status: ExitStatus) -> Termination {
        if self.stderr_done.try_recv().is_ok()
            && let Some(stderr_reader) = self.stderr_reader.take()
        {
            let _ = stderr_reader.join();
        }
        let termination = Termination {
            status,
            stderr: self.stderr_snapshot(),
        };
        self.process_group.finish_retirement_after_wait();
        self.child.take();
        self.termination = Some(termination.clone());
        // The process-group leader has been reaped and every member received
        // SIGKILL under the PGID gate. Delegate potentially expensive recursive
        // scratch cleanup rather than extending the caller's deadline.
        self.handoff_background_reaper();
        termination
    }

    fn handoff_background_reaper(&mut self) {
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        if self.child.is_some() {
            // Register before relinquishing the child, not in the detached
            // thread. A constructor concurrent with handoff must observe the
            // retirement before this method can return.
            self.process_group.begin_retirement();
        }
        let child = self.child.take();
        let stderr_reader = self.stderr_reader.take();
        if child.is_none() && stderr_reader.is_none() && self.scratch.is_none() {
            return;
        }
        let resources = ReapResources::new(
            child,
            self.process_group.clone(),
            stderr_reader,
            self.scratch.take(),
        );
        #[cfg(test)]
        let resources = {
            let mut resources = resources;
            resources.test_barrier = self.reaper_test_barrier.take();
            resources
        };
        let handed_off = match self.background_reaper.as_mut() {
            Some(reaper) => reaper.handoff(resources),
            None => Err(resources),
        };
        if let Err(resources) = handed_off {
            // The receiver is created before the child and performs no
            // fallible work before recv. Preserve ownership even if that
            // invariant is broken instead of panicking from `Drop`.
            reap_resources(resources);
        }
    }

    fn stderr_snapshot(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| tail.render())
            .unwrap_or_else(|_| "[stderr capture unavailable]".to_string())
    }
}

#[derive(Debug)]
struct ProcessGroupDeadline {
    process_group: Arc<ManagedProcessGroup>,
    cancel: Option<mpsc::Sender<()>>,
    watchdog: Option<JoinHandle<()>>,
    timed_out: Arc<AtomicBool>,
    started: Instant,
    timeout: Duration,
}

impl ProcessGroupDeadline {
    fn arm(process_group: Arc<ManagedProcessGroup>, timeout: Duration) -> io::Result<Self> {
        let started = Instant::now();
        let (cancel, cancelled) = mpsc::channel();
        let timed_out = Arc::new(AtomicBool::new(false));
        let watchdog_timed_out = timed_out.clone();
        let watchdog_process_group = process_group.clone();
        let watchdog = thread::Builder::new()
            .name("catena-safe-runtime-watchdog".to_string())
            .spawn(move || match cancelled.recv_timeout(timeout) {
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    watchdog_timed_out.store(true, Ordering::Release);
                    watchdog_process_group.kill();
                }
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
            })?;
        Ok(Self {
            process_group,
            cancel: Some(cancel),
            watchdog: Some(watchdog),
            timed_out,
            started,
            timeout,
        })
    }

    fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Acquire) || self.started.elapsed() >= self.timeout
    }

    fn expire_now(&self) {
        self.timed_out.store(true, Ordering::Release);
        self.process_group.kill();
    }

    fn remaining(&self) -> Option<Duration> {
        if self.timed_out.load(Ordering::Acquire) {
            return None;
        }
        self.timeout.checked_sub(self.started.elapsed())
    }

    fn ensure_io_time_remaining(&self) -> io::Result<()> {
        if self.remaining().is_none() {
            self.timed_out.store(true, Ordering::Release);
            Err(protocol_timeout_error())
        } else {
            Ok(())
        }
    }

    fn wait_for_descriptor(&self, descriptor: RawFd, events: libc::c_short) -> io::Result<()> {
        loop {
            let Some(remaining) = self.remaining() else {
                self.timed_out.store(true, Ordering::Release);
                return Err(protocol_timeout_error());
            };
            let fractional_millisecond = remaining.subsec_nanos() % 1_000_000 != 0;
            let timeout_millis = remaining
                .as_millis()
                .saturating_add(u128::from(fractional_millisecond))
                .clamp(1, i32::MAX as u128) as i32;
            let mut descriptor = libc::pollfd {
                fd: descriptor,
                events,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut descriptor, 1, timeout_millis) };
            if result > 0 {
                if descriptor.revents & libc::POLLNVAL != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "SafeRuntime protocol descriptor became invalid",
                    ));
                }
                // POLLHUP/POLLERR are deliberately treated as readiness: the
                // following read/write reports EOF or the precise pipe error.
                return Ok(());
            }
            if result == 0 {
                continue;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    fn finish(mut self) -> bool {
        let elapsed = self.timed_out();
        self.cancel();
        elapsed || self.timed_out.load(Ordering::Acquire)
    }

    fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        if let Some(watchdog) = self.watchdog.take()
            && watchdog.join().is_err()
        {
            self.timed_out.store(true, Ordering::Release);
        }
    }
}

fn protocol_timeout_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "SafeRuntime protocol deadline elapsed",
    )
}

impl Drop for ProcessGroupDeadline {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.termination.is_none() && self.child.is_some() {
            // Close admission before graceful shutdown or SIGKILL. Drop stays
            // bounded: this is one idempotent atomic registration, never a wait
            // for constructors that linearized before it.
            self.process_group.begin_retirement();
            #[cfg(test)]
            if let Some(barrier) = self.drop_retirement_test_barrier.take() {
                let _ = barrier.entered.send(());
                let _ = barrier.release.recv();
            }
            if self.stdin.is_some()
                && let Ok(deadline) =
                    ProcessGroupDeadline::arm(self.process_group.clone(), DROP_GRACE_TIMEOUT)
            {
                let _ = self.send(&Request::Shutdown, &deadline);
                self.stdin.take();
                // Give the worker a bounded chance to run destructors
                // without reaping its leader. The unreaped PID keeps the
                // PGID from being reused; group cleanup remains mandatory.
                let descriptor = self.stdout.get_ref().as_raw_fd();
                let _ = deadline.wait_for_descriptor(descriptor, libc::POLLIN);
                let _ = deadline.finish();
            }
            self.stdin.take();
            let _ = self.asset_socket.shutdown(Shutdown::Both);
            self.process_group.kill();
            let status = self.wait_for_exit(KILL_REAP_TIMEOUT).ok().flatten();
            if let Some(status) = status {
                self.record_exit(status);
            } else {
                self.handoff_background_reaper();
            }
        }
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        if self.stderr_reader.is_some() {
            self.handoff_background_reaper();
        }
    }
}

fn map_init_worker_error(error: WorkerError) -> SafeInitError {
    match error {
        WorkerError::ReentrantOperation => SafeInitError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => SafeInitError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut {
            kind: DeadlineKind::Compile,
            timeout,
        } => SafeInitError::CompileTimedOut { timeout },
        WorkerError::TimedOut {
            kind: DeadlineKind::Initialize,
            timeout,
        } => SafeInitError::InitializationTimedOut { timeout },
        WorkerError::Terminated(termination) => SafeInitError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeInitError::Transport(other.to_string()),
    }
}

fn kill_process_group(process_group: u32) {
    let Ok(process_group) = i32::try_from(process_group) else {
        return;
    };
    // WorkerProcess::spawn makes the child PID its process-group ID. Killing
    // the negative ID stops that child and all compiler descendants. ESRCH is
    // benign: the process may have exited at the deadline boundary.
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
}

fn map_exec_worker_error(error: WorkerError) -> SafeExecError {
    match error {
        WorkerError::ReentrantOperation => SafeExecError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => SafeExecError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => SafeExecError::TimedOut { timeout },
        WorkerError::Terminated(termination) => SafeExecError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeExecError::Transport(other.to_string()),
    }
}

fn map_asset_worker_error(error: WorkerError) -> AssetError {
    match error {
        WorkerError::ReentrantOperation => AssetError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => AssetError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => AssetError::TimedOut { timeout },
        WorkerError::Terminated(termination) => AssetError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => AssetError::Transport(other.to_string()),
    }
}

fn map_resident_worker_error(error: WorkerError) -> ResidentError {
    match error {
        WorkerError::ReentrantOperation => ResidentError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => ResidentError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => ResidentError::TimedOut { timeout },
        WorkerError::Terminated(termination) => ResidentError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => ResidentError::Transport(other.to_string()),
    }
}

fn map_resident_teardown_worker_error(error: WorkerError) -> ResidentError {
    match error {
        WorkerError::LocalRequest(error) => ResidentError::Teardown(error.to_string()),
        other => map_resident_worker_error(other),
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    const TIMEOUT_FIXTURE_ENV: &str = "CATENA_TEST_PROCESS_GROUP_TIMEOUT_FIXTURE";
    const DROP_FIXTURE_ENV: &str = "CATENA_TEST_BOUNDED_DROP_FIXTURE";
    const EOF_FIXTURE_ENV: &str = "CATENA_TEST_UNEXPECTED_EOF_FIXTURE";
    const EOF_DESCENDANT_ENV: &str = "CATENA_TEST_UNEXPECTED_EOF_DESCENDANT";
    const SCRATCH_FIXTURE_ENV: &str = "CATENA_TEST_SCRATCH_FIXTURE";
    const SCRATCH_DESCENDANT_ENV: &str = "CATENA_TEST_SCRATCH_DESCENDANT";

    fn target_backed_test_directory(prefix: &str) -> tempfile::TempDir {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("catena-lang crate must live in its workspace");
        let parent = workspace.join("target/safe-runtime-lifecycle-tests");
        fs::create_dir_all(&parent).expect("create target-backed lifecycle test directory");
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in(parent)
            .expect("create target-backed lifecycle fixture")
    }

    fn wait_for_path_state(path: &Path, exists: bool, message: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while path.exists() != exists {
            assert!(Instant::now() < deadline, "{message}: {}", path.display());
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn structured_rejections_preserve_the_session() {
        assert!(!SafeInitError::RemoteLoad("invalid source".into()).invalidates_session());
        assert!(!SafeInitError::InvalidRequest("oversized".into()).invalidates_session());
        assert!(!AssetError::Remote("invalid asset".into()).invalidates_session());
        assert!(!AssetError::InvalidRequest("oversized".into()).invalidates_session());
        assert!(!ResidentError::Remote("invalid request".into()).invalidates_session());
        assert!(!ResidentError::InvalidRequest("oversized".into()).invalidates_session());
        assert!(!SafeExecError::UnknownArtifact.invalidates_session());
        assert!(!SafeExecError::RemoteMemory("invalid view".into()).invalidates_session());
        assert!(!SafeExecError::InvalidRequest("oversized".into()).invalidates_session());
    }

    #[test]
    fn protocol_failures_invalidate_the_session() {
        assert!(SafeInitError::UnexpectedResponse.invalidates_session());
        assert!(
            SafeInitError::CompileTimedOut {
                timeout: Duration::from_secs(1)
            }
            .invalidates_session()
        );
        assert!(ResidentError::UnexpectedResponse.invalidates_session());
        assert!(ResidentError::Teardown("missing acknowledgement".into()).invalidates_session());
        assert!(SafeExecError::UnexpectedResponse.invalidates_session());
        assert!(SafeExecError::Transport("partial write".into()).invalidates_session());
    }

    #[test]
    fn generation_watchdog_never_overlaps_a_queued_session_operation() {
        let gate = Arc::new(SessionOperationGate::default());
        let generation = gate
            .acquire(SessionOperationKind::Generation)
            .expect("acquire generation fixture");
        let process_group = Arc::new(ManagedProcessGroup::new_test(u32::MAX));
        let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_millis(40))
            .expect("arm generation fixture watchdog");
        let (attempted_sender, attempted) = mpsc::channel();
        let (acquired_sender, acquired) = mpsc::channel();

        thread::scope(|scope| {
            let contender_gate = gate.clone();
            scope.spawn(move || {
                attempted_sender.send(()).expect("announce gate attempt");
                let _operation = contender_gate
                    .acquire(SessionOperationKind::Ordinary)
                    .expect("queued operation must eventually acquire the gate");
                acquired_sender.send(()).expect("announce gate acquisition");
            });
            attempted
                .recv_timeout(Duration::from_secs(1))
                .expect("contending operation did not start");

            let watchdog_limit = Instant::now() + Duration::from_secs(1);
            while !deadline.timed_out.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < watchdog_limit,
                    "generation watchdog did not fire"
                );
                thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(process_group.kill_count(), 1);
            assert!(
                matches!(acquired.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "an unrelated operation entered while the generation watchdog was armed"
            );

            drop(generation);
            acquired
                .recv_timeout(Duration::from_secs(1))
                .expect("queued operation was not released with the generation lease");
        });
        assert!(deadline.finish());
    }

    #[test]
    fn same_thread_generation_reentry_is_rejected_and_model_drop_is_deferred() {
        let gate = Arc::new(SessionOperationGate::default());
        let generation = gate
            .acquire(SessionOperationKind::Generation)
            .expect("acquire generation fixture");

        assert!(
            gate.acquire(SessionOperationKind::Ordinary).is_err(),
            "same-thread callback re-entry must not deadlock"
        );
        assert!(
            gate.acquire_or_defer_model_release(17).is_err(),
            "same-thread model release cannot wait during uncommitted startup"
        );
        generation.open_model_release_deferrals();
        assert!(matches!(gate.acquire_or_defer_model_release(17), Ok(None)));
        assert!(matches!(gate.acquire_or_defer_model_release(23), Ok(None)));
        assert_eq!(
            generation.seal_model_release_deferrals(),
            vec![DeferredRelease::Model(17), DeferredRelease::Model(23)]
        );
        drop(generation);

        let _next = gate
            .acquire(SessionOperationKind::Ordinary)
            .expect("gate must be reusable after generation teardown");
    }

    #[test]
    fn artifact_release_from_callback_is_deferred_and_requires_matching_ack() {
        let gate = Arc::new(SessionOperationGate::default());
        let generation = gate.acquire(SessionOperationKind::Generation).unwrap();
        generation.open_model_release_deferrals();
        let release = DeferredRelease::Artifact(7);
        assert!(matches!(gate.acquire_or_defer_release(release), Ok(None)));
        assert_eq!(generation.seal_model_release_deferrals(), vec![release]);
        assert!(matches!(
            release.request(),
            Request::ReleaseArtifact { artifact: 7 }
        ));
        assert!(
            release.acknowledged(&Response::Resident(Ok(ResidentResponse::ArtifactReleased(
                7
            ),)))
        );
        assert!(!release.acknowledged(&Response::Resident(Ok(
            ResidentResponse::ArtifactReleased(8),
        ))));
        assert!(
            !release.acknowledged(&Response::Resident(Ok(ResidentResponse::ModelReleased(7),)))
        );
        drop(generation);
        let _next = gate.acquire(SessionOperationKind::Ordinary).unwrap();
    }

    #[test]
    fn preopen_model_release_defers_after_start_and_waits_after_seal() {
        let gate = Arc::new(SessionOperationGate::default());
        let generation = gate
            .acquire(SessionOperationKind::Generation)
            .expect("acquire generation fixture");

        thread::scope(|scope| {
            let callback_drop_gate = gate.clone();
            let (completed_sender, completed) = mpsc::sync_channel(1);
            scope.spawn(move || {
                let release = callback_drop_gate
                    .acquire_or_defer_model_release(17)
                    .expect("cross-thread model release admission");
                completed_sender
                    .send((17, release.is_none()))
                    .expect("report pre-open model drop outcome");
            });
            gate.wait_for_model_release_waiter();
            assert!(
                matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "model release completed before generation startup committed"
            );

            generation.open_model_release_deferrals();
            assert_eq!(
                completed
                    .recv_timeout(Duration::from_secs(1))
                    .expect("pre-open model release did not join deferral after start"),
                (17, true)
            );
            assert_eq!(
                generation.seal_model_release_deferrals(),
                vec![DeferredRelease::Model(17)]
            );

            let post_seal_gate = gate.clone();
            let (completed_sender, completed) = mpsc::sync_channel(1);
            scope.spawn(move || {
                let release = post_seal_gate
                    .acquire_or_defer_model_release(23)
                    .expect("post-seal model release admission");
                completed_sender
                    .send((23, release.is_none()))
                    .expect("report post-seal model drop outcome");
            });
            gate.wait_for_model_release_waiter();
            assert!(
                matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "post-seal model release entered before generation teardown"
            );

            drop(generation);
            assert_eq!(
                completed
                    .recv_timeout(Duration::from_secs(1))
                    .expect("post-seal model release did not resume after generation teardown"),
                (23, false)
            );
        });
    }

    #[test]
    fn preopen_model_release_acquires_after_failed_generation_start() {
        let gate = Arc::new(SessionOperationGate::default());
        let generation = gate
            .acquire(SessionOperationKind::Generation)
            .expect("acquire generation-start fixture");

        thread::scope(|scope| {
            let model_drop_gate = gate.clone();
            let (completed_sender, completed) = mpsc::sync_channel(1);
            scope.spawn(move || {
                let release = model_drop_gate
                    .acquire_or_defer_model_release(29)
                    .expect("failed-start model release admission");
                completed_sender
                    .send((29, release.is_none()))
                    .expect("report failed-start model drop outcome");
            });
            gate.wait_for_model_release_waiter();
            assert!(
                matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "model release completed before generation startup resolved"
            );

            // Dropping a still-closed lease models every StartGeneration error.
            // The waiting destructor must retain its model ID and acquire the
            // ordinary release lease rather than disappear into a stale queue.
            drop(generation);
            assert_eq!(
                completed
                    .recv_timeout(Duration::from_secs(1))
                    .expect("model release did not resume after failed startup"),
                (29, false)
            );
        });

        let _next = gate
            .acquire(SessionOperationKind::Ordinary)
            .expect("failed-start model release did not relinquish the gate");
    }

    #[test]
    fn descriptor_transport_failure_is_a_fatal_child_protocol_error() {
        let (mut sender, receiver) = UnixStream::pair().expect("descriptor fixture pair");
        sender
            .write_all(&[0])
            .expect("write malformed descriptor message");

        assert!(matches!(
            receive_asset_file(&receiver),
            Err(ChildMainError::Protocol(message))
                if message.contains("asset descriptor transport failed")
        ));
    }

    #[test]
    fn worker_asset_socket_duplicates_instead_of_claiming_the_supplied_descriptor() {
        let (original, mut peer) = UnixStream::pair().expect("asset socket fixture pair");
        let original_fd = original.as_raw_fd();
        let flags = unsafe { libc::fcntl(original_fd, libc::F_GETFD) };
        assert_ne!(flags, -1);
        assert_ne!(
            unsafe { libc::fcntl(original_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) },
            -1
        );
        let mut duplicate = duplicate_unix_stream(original_fd).expect("duplicate asset socket");
        let original_flags = unsafe { libc::fcntl(original_fd, libc::F_GETFD) };
        assert_ne!(original_flags & libc::FD_CLOEXEC, 0);
        let duplicate_flags = unsafe { libc::fcntl(duplicate.as_raw_fd(), libc::F_GETFD) };
        assert_ne!(duplicate_flags & libc::FD_CLOEXEC, 0);
        drop(original);

        duplicate.write_all(b"x").expect("write through duplicate");
        let mut byte = [0_u8; 1];
        peer.read_exact(&mut byte).expect("read duplicate payload");
        assert_eq!(byte, *b"x");
        assert_eq!(
            duplicate_unix_stream(-1)
                .expect_err("invalid descriptor must be rejected")
                .raw_os_error(),
            Some(libc::EBADF)
        );
    }

    #[test]
    fn sigchld_must_leave_children_waitable() {
        assert!(sigchld_is_waitable(libc::SIG_DFL, 0));
        assert!(!sigchld_is_waitable(libc::SIG_IGN, 0));
        assert!(!sigchld_is_waitable(libc::SIG_DFL, libc::SA_NOCLDWAIT));
    }

    #[test]
    fn zero_compile_timeout_is_rejected_before_spawning() {
        assert!(matches!(
            SafeRuntime::with_compile_timeout(GpuDialect::Hip, Duration::ZERO),
            Err(SafeInitError::InvalidCompileTimeout)
        ));
    }

    #[test]
    fn zero_execution_timeout_is_rejected_before_spawning() {
        let timeouts = SafeRuntimeTimeouts::default().with_execution_timeout(Duration::ZERO);
        assert!(matches!(
            SafeRuntime::with_timeouts(GpuDialect::Hip, timeouts),
            Err(SafeInitError::InvalidExecutionTimeout)
        ));
    }

    #[test]
    fn resident_protocol_construction_does_not_initialize_parent_ipc() {
        let calls = std::cell::Cell::new(0);
        let ipc = initialize_parent_ipc(ParentInterface::ResidentProtocol, || {
            calls.set(calls.get() + 1);
            Ok::<_, std::convert::Infallible>(())
        })
        .expect("infallible parent IPC fixture");

        assert!(ipc.is_none());
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn raw_value_construction_initializes_parent_ipc() {
        let calls = std::cell::Cell::new(0);
        let ipc = initialize_parent_ipc(ParentInterface::RawValues, || {
            calls.set(calls.get() + 1);
            Ok::<_, std::convert::Infallible>("parent IPC")
        })
        .expect("infallible parent IPC fixture");

        assert_eq!(ipc, Some("parent IPC"));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn stderr_capture_retains_only_a_marked_fixed_tail() {
        let mut tail = StderrTail::default();
        tail.push(&vec![b'x'; STDERR_TAIL_BYTES + 17]);
        tail.push(b"final marker");

        assert_eq!(tail.bytes.len(), STDERR_TAIL_BYTES);
        let rendered = tail.render();
        assert!(rendered.starts_with("[... "));
        assert!(rendered.contains("stderr truncated"));
        assert!(rendered.ends_with("final marker"));
    }

    #[test]
    fn compile_deadline_kills_the_worker_process_group() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::compile_deadline_process_fixture",
            ])
            .env(TIMEOUT_FIXTURE_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = command.spawn().expect("spawn timeout fixture");
        let process_group = Arc::new(ManagedProcessGroup::new_test(child.id()));
        let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_millis(50))
            .expect("spawn deadline watchdog");
        let wait_deadline = Instant::now() + Duration::from_secs(2);
        while !deadline.timed_out.load(Ordering::Acquire) {
            assert!(
                Instant::now() < wait_deadline,
                "deadline watchdog did not fire"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let status = loop {
            if let Some(status) = process_group
                .kill_and_try_wait(&mut child)
                .expect("poll timeout fixture")
            {
                break status;
            }
            assert!(
                Instant::now() < wait_deadline,
                "deadline did not stop its process fixture"
            );
            thread::sleep(Duration::from_millis(5));
        };
        assert!(deadline.finish(), "fixture exited before its deadline");
        assert!(!status.success(), "deadline did not kill the fixture");
    }

    #[test]
    fn protocol_pipe_read_and_write_share_an_absolute_deadline() {
        let (mut reader, mut held_writer) = UnixStream::pair().expect("reader fixture pair");
        held_writer
            .write_all(&[1, 0])
            .expect("write partial frame header");
        set_nonblocking(reader.as_raw_fd()).expect("nonblocking reader fixture");
        let read_deadline = ProcessGroupDeadline::arm(
            Arc::new(ManagedProcessGroup::new_test(u32::MAX)),
            Duration::from_millis(40),
        )
        .expect("spawn read deadline watchdog");
        let descriptor = reader.as_raw_fd();
        let started = Instant::now();
        let error = read_frame::<Request>(&mut DeadlineReader::new(
            &mut reader,
            descriptor,
            &read_deadline,
        ))
        .expect_err("partial frame should reach its deadline");
        assert!(matches!(
            error,
            ProtocolError::Io(error) if error.kind() == io::ErrorKind::TimedOut
        ));
        assert!(read_deadline.finish());
        assert!(started.elapsed() < Duration::from_secs(1));

        let (mut writer, _held_reader) = UnixStream::pair().expect("writer fixture pair");
        set_nonblocking(writer.as_raw_fd()).expect("nonblocking writer fixture");
        let write_deadline = ProcessGroupDeadline::arm(
            Arc::new(ManagedProcessGroup::new_test(u32::MAX)),
            Duration::from_millis(40),
        )
        .expect("spawn write deadline watchdog");
        let descriptor = writer.as_raw_fd();
        let started = Instant::now();
        let error = write_frame(
            &mut DeadlineWriter::new(&mut writer, descriptor, &write_deadline),
            &Request::LoadSources {
                sources: vec!["x".repeat(8 * 1024 * 1024)],
            },
        )
        .expect_err("undrained partial frame should reach its deadline");
        assert!(matches!(
            error,
            ProtocolError::Io(error) if error.kind() == io::ErrorKind::TimedOut
        ));
        assert!(write_deadline.finish());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn worker_drop_is_bounded_when_the_child_ignores_shutdown() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let worker =
            WorkerProcess::spawn_command(command, &executable).expect("spawn bounded-drop fixture");
        thread::sleep(Duration::from_millis(20));
        let process = worker.child_id().expect("worker process ID");
        assert_eq!(
            unsafe { libc::kill(i32::try_from(process).unwrap(), 0) },
            0,
            "drop fixture exited before WorkerProcess was dropped"
        );

        let started = Instant::now();
        drop(worker);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "WorkerProcess::drop exceeded its grace and reap bounds"
        );
    }

    #[test]
    fn forced_worker_termination_repeatedly_cleans_parent_owned_scratch() {
        let scratch_parent = target_backed_test_directory("scratch-parent-");
        let executable = env::current_exe().expect("test executable path");

        for teardown in ["watchdog", "reaper", "drop", "watchdog", "reaper", "drop"] {
            let mut command = Command::new(&executable);
            command
                .args([
                    "--ignored",
                    "--exact",
                    "safe_runtime::error_tests::scratch_process_fixture",
                    "--nocapture",
                ])
                .env(SCRATCH_FIXTURE_ENV, "1");
            let mut worker =
                WorkerProcess::spawn_command_in(command, &executable, scratch_parent.path())
                    .expect("spawn scratch fixture");
            let scratch = worker
                .scratch
                .as_ref()
                .expect("worker owns scratch")
                .path()
                .to_path_buf();
            wait_for_path_state(
                &scratch.join("ready"),
                true,
                "scratch fixture did not become ready",
            );

            match teardown {
                "watchdog" => {
                    let result: Result<(), WorkerError> = worker.with_deadline(
                        Duration::from_millis(25),
                        DeadlineKind::Compile,
                        |_worker, _deadline| {
                            thread::sleep(Duration::from_millis(75));
                            Ok(())
                        },
                    );
                    assert!(matches!(result, Err(WorkerError::TimedOut { .. })));
                    drop(worker);
                }
                "reaper" => {
                    worker.handoff_background_reaper();
                    drop(worker);
                }
                "drop" => drop(worker),
                _ => unreachable!(),
            }

            wait_for_path_state(
                &scratch,
                false,
                "forced worker teardown leaked its scratch directory",
            );
        }
    }

    #[test]
    fn pathological_worker_resources_are_handed_to_a_background_reaper() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let mut worker =
            WorkerProcess::spawn_command(command, &executable).expect("spawn reap fixture");
        let process = worker.child_id().expect("worker process ID");
        assert!(
            worker
                .background_reaper
                .as_ref()
                .is_some_and(|reaper| reaper.handle.is_some()),
            "reaper must exist before teardown begins"
        );
        worker.handoff_background_reaper();
        assert!(worker.child.is_none());
        assert!(worker.stderr_reader.is_none());

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let result = unsafe { libc::kill(i32::try_from(process).unwrap(), 0) };
            if result == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "background reaper did not wait for the killed worker"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn retirement_gate_counts_callers_and_resets_in_a_forked_process() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let parent = 41;
        let already_admitted = gate
            .begin_spawn_for(parent)
            .expect("initial spawn must be admitted");
        let first = gate.begin_retirement_for(parent);
        let second = gate.begin_retirement_for(parent);
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 2 })
        ));
        drop(already_admitted);
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 2 })
        ));
        first.finish_after_wait();
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));

        // A forked child copied the atomic word, but none of the parent's
        // workers are its children. Its PID starts a fresh lifecycle epoch.
        let child = 42;
        let child_spawn = gate
            .begin_spawn_for(child)
            .expect("forked process must reset copied retirement state");
        drop(child_spawn);
        second.finish_after_wait();
        let child_spawn = gate
            .begin_spawn_for(child)
            .expect("parent permit must not alter child epoch");
        drop(child_spawn);
    }

    #[test]
    fn dropped_retirement_permit_keeps_worker_admission_closed() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let process = 41;
        let retirement = gate.begin_retirement_for(process);
        drop(retirement);
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));
    }

    #[test]
    fn retirement_saturation_is_sticky_after_tracked_completions() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let process = 41;
        gate.state.store(
            WorkerLifecycleGate::encode(process, 0, LIFECYCLE_RETIREMENT_SATURATED - 2),
            Ordering::Release,
        );

        let last_tracked = gate.begin_retirement_for(process);
        assert!(last_tracked.completable);
        assert_eq!(
            WorkerLifecycleGate::decode(gate.state.load(Ordering::Acquire)).2,
            LIFECYCLE_RETIREMENT_SATURATED - 1
        );

        let saturated = gate.begin_retirement_for(process);
        assert!(!saturated.completable);
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped {
                workers: LIFECYCLE_RETIREMENT_SATURATED
            })
        ));

        last_tracked.finish_after_wait();
        saturated.finish_after_wait();
        assert_eq!(
            WorkerLifecycleGate::decode(gate.state.load(Ordering::Acquire)).2,
            LIFECYCLE_RETIREMENT_SATURATED
        );
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped {
                workers: LIFECYCLE_RETIREMENT_SATURATED
            })
        ));
    }

    #[test]
    fn every_detached_teardown_gates_replacement_until_reap() {
        for teardown in ["handoff", "watchdog", "drop"] {
            let executable = env::current_exe().expect("test executable path");
            let mut command = Command::new(&executable);
            command
                .args([
                    "--ignored",
                    "--exact",
                    "safe_runtime::error_tests::bounded_drop_process_fixture",
                ])
                .env(DROP_FIXTURE_ENV, "1");
            let gate = Arc::new(WorkerLifecycleGate::default());
            let mut worker =
                WorkerProcess::spawn_command_with_gate(command, &executable, gate.clone())
                    .expect("spawn retirement-gate fixture");
            let (entered_sender, entered) = mpsc::sync_channel(1);
            let (release, release_receiver) = mpsc::channel();
            worker.force_background_handoff = true;
            worker.reaper_test_barrier = Some(Box::new(LifecycleTestBarrier {
                entered: entered_sender,
                release: release_receiver,
            }));

            match teardown {
                "handoff" => worker.handoff_background_reaper(),
                "watchdog" => {
                    let result: Result<(), WorkerError> = worker.with_deadline(
                        Duration::from_millis(20),
                        DeadlineKind::Execution,
                        |_worker, _deadline| {
                            thread::sleep(Duration::from_millis(60));
                            Ok(())
                        },
                    );
                    assert!(matches!(result, Err(WorkerError::TimedOut { .. })));
                }
                "drop" => drop(worker),
                _ => unreachable!(),
            }

            entered
                .recv_timeout(Duration::from_secs(2))
                .unwrap_or_else(|_| panic!("{teardown} did not hand its leader to the reaper"));
            assert!(matches!(
                gate.begin_spawn(),
                Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
            ));
            release
                .send(())
                .expect("release retirement-gate fixture reaper");

            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match gate.begin_spawn() {
                    Ok(permit) => {
                        drop(permit);
                        break;
                    }
                    Err(SafeInitError::PreviousWorkerUnreaped { .. }) => {
                        assert!(
                            Instant::now() < deadline,
                            "{teardown} did not reopen worker admission after wait"
                        );
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("unexpected lifecycle error after {teardown}: {error}"),
                }
            }
        }
    }

    #[test]
    fn drop_registers_retirement_before_graceful_shutdown() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let gate = Arc::new(WorkerLifecycleGate::default());
        let mut worker = WorkerProcess::spawn_command_with_gate(command, &executable, gate.clone())
            .expect("spawn pre-handoff retirement fixture");
        let (entered_sender, entered) = mpsc::sync_channel(1);
        let (release, release_receiver) = mpsc::channel();
        worker.drop_retirement_test_barrier = Some(Box::new(LifecycleTestBarrier {
            entered: entered_sender,
            release: release_receiver,
        }));

        thread::scope(|scope| {
            let dropper = scope.spawn(move || drop(worker));
            entered
                .recv_timeout(Duration::from_secs(2))
                .expect("Drop did not register retirement before graceful shutdown");
            assert!(matches!(
                gate.begin_spawn(),
                Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
            ));
            release
                .send(())
                .expect("release pre-handoff retirement fixture");
            dropper.join().expect("Drop fixture thread panicked");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match gate.begin_spawn() {
                Ok(permit) => {
                    drop(permit);
                    break;
                }
                Err(SafeInitError::PreviousWorkerUnreaped { .. }) => {
                    assert!(
                        Instant::now() < deadline,
                        "Drop did not reopen admission after foreground or detached wait"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("unexpected lifecycle error after Drop: {error}"),
            }
        }
    }

    #[test]
    fn unexpected_eof_kills_the_worker_process_group_before_reaping() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::unexpected_eof_process_fixture",
                "--nocapture",
            ])
            .env(EOF_FIXTURE_ENV, "1");
        let mut worker = WorkerProcess::spawn_command(command, &executable)
            .expect("spawn unexpected-EOF fixture");
        let deadline = worker
            .arm_deadline(Duration::from_secs(5))
            .expect("arm EOF fixture deadline");

        // Drain test-harness chatter until the worker leader closes its
        // protocol pipe. Its sleeping descendant deliberately retains stderr.
        let descriptor = worker.stdout.get_ref().as_raw_fd();
        let mut output = Vec::new();
        let mut bytes = [0_u8; 1024];
        loop {
            let read = DeadlineReader::new(&mut worker.stdout, descriptor, &deadline)
                .read(&mut bytes)
                .expect("drain EOF fixture stdout");
            if read == 0 {
                break;
            }
            output.extend_from_slice(&bytes[..read]);
        }
        assert!(
            String::from_utf8_lossy(&output).contains("CATENA_DESCENDANT_STARTED"),
            "EOF fixture did not start its inherited process-group descendant"
        );
        loop {
            let ready = worker
                .stderr_tail
                .lock()
                .expect("lock EOF fixture stderr")
                .render()
                .contains("CATENA_DESCENDANT_READY");
            if ready {
                break;
            }
            deadline
                .ensure_io_time_remaining()
                .expect("descendant did not become ready before EOF cleanup deadline");
            thread::sleep(Duration::from_millis(5));
        }

        let termination = worker.reap(&deadline).expect("reap EOF fixture");
        assert!(!termination.status.success());
        assert!(worker.process_group.kill_count() >= 1);
        assert!(!deadline.finish(), "EOF cleanup exceeded its deadline");

        // The descendant inherited the worker's stderr descriptor and sleeps
        // for a minute. EOF here therefore proves group cleanup stopped it.
        if let Some(stderr_reader) = worker.stderr_reader.take() {
            worker
                .stderr_done
                .recv_timeout(Duration::from_secs(2))
                .expect("worker descendant retained stderr after group cleanup");
            stderr_reader
                .join()
                .expect("join EOF fixture stderr reader");
        }
    }

    #[test]
    fn stale_watchdog_cannot_signal_after_the_group_leader_is_reaped() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(executable);
        command
            .arg("--list")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = command.spawn().expect("spawn short-lived group leader");
        let process_group = Arc::new(ManagedProcessGroup::new_test(child.id()));
        let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_secs(2))
            .expect("spawn stale deadline watchdog");
        let wait_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if process_group
                .kill_and_try_wait(&mut child)
                .expect("poll short-lived group leader")
                .is_some()
            {
                break;
            }
            assert!(
                Instant::now() < wait_deadline,
                "short-lived group leader did not exit before its watchdog"
            );
            thread::sleep(Duration::from_millis(5));
        }

        let watchdog_deadline = Instant::now() + Duration::from_secs(3);
        while !deadline.timed_out.load(Ordering::Acquire) {
            assert!(
                Instant::now() < watchdog_deadline,
                "stale watchdog did not fire"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let kills_at_reap = process_group.kill_count();
        assert!(deadline.finish());
        assert_eq!(
            process_group.kill_count(),
            kills_at_reap,
            "retired process-group identity was signalled by a stale watchdog"
        );
    }

    #[test]
    fn protocol_failure_automatically_poisons_the_worker() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let mut worker =
            WorkerProcess::spawn_command(command, &executable).expect("spawn poison fixture");
        let result: Result<(), WorkerError> = worker.with_deadline(
            Duration::from_secs(1),
            DeadlineKind::Execution,
            |_worker, _deadline| {
                Err(WorkerError::Protocol(ProtocolError::Io(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "fixture transport failure",
                ))))
            },
        );

        assert!(matches!(result, Err(WorkerError::Protocol(_))));
        assert!(worker.stdin.is_none());
        assert!(worker.child.is_none());
        assert!(matches!(
            worker.ensure_available(),
            Err(WorkerError::Invalidated)
        ));
    }

    #[test]
    fn local_request_preparation_failure_preserves_the_worker() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let mut worker = WorkerProcess::spawn_command(command, &executable)
            .expect("spawn local-request fixture");
        let process = worker.child_id().expect("worker process ID");
        let result: Result<(), WorkerError> = worker.with_deadline(
            Duration::from_secs(1),
            DeadlineKind::Execution,
            |_worker, _deadline| {
                Err(WorkerError::LocalRequest(FrameEncodeError::FrameTooLarge {
                    actual: protocol::MAX_FRAME_LEN + 1,
                    maximum: protocol::MAX_FRAME_LEN,
                }))
            },
        );

        assert!(matches!(result, Err(WorkerError::LocalRequest(_))));
        assert_eq!(worker.child_id().expect("worker remains owned"), process);
        assert!(worker.ensure_available().is_ok());
    }

    #[test]
    fn structured_memory_rejection_preserves_the_worker() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let mut worker = WorkerProcess::spawn_command(command, &executable)
            .expect("spawn structured-memory-rejection fixture");
        let process = worker.child_id().expect("worker process ID");

        let error = execution_from_response(
            &mut worker,
            Response::Executed(Err(RemoteExecError::Memory("invalid IPC view".into()))),
        )
        .expect_err("structured memory rejection must remain an error");

        assert!(matches!(error, SafeExecError::RemoteMemory(_)));
        assert_eq!(worker.child_id().expect("worker remains owned"), process);
        assert!(worker.ensure_available().is_ok());

        let error = execution_from_response(
            &mut worker,
            Response::Executed(Err(RemoteExecError::Protocol("pending outputs".into()))),
        )
        .expect_err("protocol-state rejection must remain an error");
        assert!(matches!(error, SafeExecError::Transport(_)));
        assert!(matches!(
            worker.ensure_available(),
            Err(WorkerError::Invalidated)
        ));
    }

    #[test]
    fn poisoned_model_teardown_lock_terminates_the_worker() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let worker = Mutex::new(
            WorkerProcess::spawn_command(command, &executable)
                .expect("spawn poisoned-model-teardown fixture"),
        );
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held_worker = worker.lock().expect("lock model-teardown fixture");
            panic!("poison model-teardown fixture");
        }));
        assert!(panicked.is_err());

        let error = worker_for_model_teardown(worker.lock())
            .expect_err("poisoned teardown lock must not yield a live worker");
        assert!(matches!(error, ResidentError::Transport(_)));
        let worker = worker
            .lock()
            .expect_err("fixture mutex remains poisoned")
            .into_inner();
        assert!(matches!(
            worker.ensure_available(),
            Err(WorkerError::Invalidated)
        ));
    }

    #[test]
    fn unexpected_response_automatically_poisons_the_worker() {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let mut worker = WorkerProcess::spawn_command(command, &executable)
            .expect("spawn unexpected-response fixture");

        let error = worker.reject_protocol_response(ResidentError::UnexpectedResponse);

        assert!(error.invalidates_session());
        assert!(worker.stdin.is_none());
        assert!(worker.child.is_none());
        assert!(matches!(
            worker.ensure_available(),
            Err(WorkerError::Invalidated)
        ));
    }

    #[test]
    #[ignore = "spawned only by compile_deadline_kills_the_worker_process_group"]
    fn compile_deadline_process_fixture() {
        assert_eq!(
            env::var_os(TIMEOUT_FIXTURE_ENV).as_deref(),
            Some(std::ffi::OsStr::new("1"))
        );
        thread::sleep(Duration::from_secs(60));
    }

    #[test]
    #[ignore = "spawned only by worker_drop_is_bounded_when_the_child_ignores_shutdown"]
    fn bounded_drop_process_fixture() {
        assert_eq!(
            env::var_os(DROP_FIXTURE_ENV).as_deref(),
            Some(std::ffi::OsStr::new("1"))
        );
        if env::var_os(EOF_DESCENDANT_ENV).is_some() {
            eprintln!("CATENA_DESCENDANT_READY");
            io::stderr().flush().expect("flush EOF descendant marker");
        }
        thread::sleep(Duration::from_secs(60));
    }

    #[test]
    #[ignore = "spawned only by unexpected_eof_kills_the_worker_process_group_before_reaping"]
    fn unexpected_eof_process_fixture() {
        assert_eq!(
            env::var_os(EOF_FIXTURE_ENV).as_deref(),
            Some(std::ffi::OsStr::new("1"))
        );
        let executable = env::current_exe().expect("test executable path");
        Command::new(executable)
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
                "--nocapture",
            ])
            .env(DROP_FIXTURE_ENV, "1")
            .env(EOF_DESCENDANT_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn inherited process-group descendant");
        println!("CATENA_DESCENDANT_STARTED");
        io::stdout().flush().expect("flush EOF fixture marker");
        std::process::exit(23);
    }

    #[test]
    #[ignore = "spawned only by forced_worker_termination_repeatedly_cleans_parent_owned_scratch"]
    fn scratch_process_fixture() {
        assert_eq!(
            env::var_os(SCRATCH_FIXTURE_ENV).as_deref(),
            Some(std::ffi::OsStr::new("1"))
        );
        let scratch = env::temp_dir();
        if env::var_os(SCRATCH_DESCENDANT_ENV).is_some() {
            let descendant = scratch.join("compiler-descendant");
            fs::create_dir_all(&descendant).expect("create compiler descendant scratch");
            fs::write(descendant.join("output.o"), b"partial compiler output")
                .expect("write compiler descendant scratch");
            thread::sleep(Duration::from_secs(60));
            return;
        }

        let worker = scratch.join("worker-build");
        fs::create_dir_all(&worker).expect("create worker scratch");
        fs::write(worker.join("module.cpp"), b"partial generated source")
            .expect("write worker scratch");
        let executable = env::current_exe().expect("test executable path");
        let mut compiler = Command::new(executable)
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::scratch_process_fixture",
                "--nocapture",
            ])
            .env(SCRATCH_FIXTURE_ENV, "1")
            .env(SCRATCH_DESCENDANT_ENV, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn compiler descendant fixture");
        wait_for_path_state(
            &scratch.join("compiler-descendant/output.o"),
            true,
            "compiler descendant did not create scratch",
        );
        fs::write(scratch.join("ready"), b"ready").expect("write scratch ready marker");
        thread::sleep(Duration::from_secs(60));
        let _ = compiler.kill();
        let _ = compiler.wait();
    }
}
