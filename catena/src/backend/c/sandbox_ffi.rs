//! Run functions from a shared object file inside an isolated helper process.
//!
//! This module is the low-level boundary between the in-process Rust API and a
//! crash-isolating worker process that owns the loaded shared object and
//! executes C functions on the caller's behalf.

use std::process::ExitStatus;
use thiserror::Error;

use super::compile::SharedObject;
use super::runtime::Value;

/// Handle to a sandbox worker process hosting one compiled shared object.
#[derive(Debug)]
pub struct SandboxProcess {
    _artifact: SharedObject,
}

/// Opaque identifier for a value owned by the sandbox worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemoteValueId(u64);

/// Errors returned by the sandbox FFI layer.
#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("Failed to spawn sandbox worker: {0}")]
    Spawn(std::io::Error),
    #[error("Sandbox worker terminated unexpectedly: {0}")]
    UnexpectedExit(ExitStatus),
    #[error("Sandbox worker protocol error: {0}")]
    Protocol(String),
    #[error("Sandbox worker does not have a loaded program")]
    NoProgramLoaded,
    #[error("Function '{0}' is not available in the loaded program")]
    UnknownFunction(String),
    #[error("Sandbox worker crashed while executing '{0}'")]
    Crashed(String),
    #[error("Unsupported value for sandbox transport")]
    UnsupportedValue,
}

impl SandboxProcess {
    /// Create a fresh sandbox worker process for one compiled shared object.
    #[allow(dead_code)]
    pub(crate) fn new(artifact: SharedObject) -> Result<Self, SandboxError> {
        let _ = &artifact;
        todo!()
    }

    /// Marshal a host value into the sandbox and obtain a remote handle.
    pub fn value(&mut self, value: Value) -> Result<RemoteValueId, SandboxError> {
        let _ = value;
        todo!()
    }

    /// Execute a function by its original Catena definition name.
    pub fn exec<const M: usize, const N: usize>(
        &mut self,
        fn_name: &str,
        args: [RemoteValueId; M],
    ) -> Result<[RemoteValueId; N], SandboxError> {
        let _ = (fn_name, args);
        todo!()
    }

    /// Discard all sandbox-owned values associated with the loaded program.
    pub fn reset(&mut self) -> Result<(), SandboxError> {
        todo!()
    }
}

impl Drop for SandboxProcess {
    fn drop(&mut self) {
        // Best-effort child shutdown belongs here once the transport exists.
    }
}

/// Entry point for the sandbox worker process.
pub fn worker_main() -> Result<(), SandboxError> {
    todo!()
}
