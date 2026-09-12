//! Session-oriented, process-isolated GPU preparation.
//!
//! [`Session`] compiles programs in one long-lived execution worker. Programs
//! keep that worker alive after the session handle is dropped.
//! A loaded program exposes its immutable entry-point metadata so an adapter can
//! reject an incompatible dynamic ABI before execution.
//!
//! Raw value execution is intentionally not exposed here. The typed
//! [`causal_lm`] interface keeps read-only assets and mutable model state in the
//! worker instead of routing model memory through a per-call transport. An
//! [`AssetOwner`] can retain immutable allocations across execution sessions.
//! Session construction loads the native GPU runtime only in workers, not
//! in the host process. The lower-level [`crate::safe_runtime`] remains
//! available for existing transient-value users that need parent-side GPU IPC.

use std::{fs::File, sync::Arc, time::Duration};

use crate::{
    runtime::RuntimeId,
    safe_runtime::{Artifact, SafeRuntime},
};

pub use crate::{
    codegen::GpuDialect,
    runtime::{Backend, EntryPoint, ValueKind},
    safe_runtime::{
        AssetError, ChildMainError, DEFAULT_COMPILE_TIMEOUT, DEFAULT_EXECUTION_TIMEOUT,
        SafeInitError, SafeRuntimeTimeouts as SessionTimeouts,
    },
};

pub mod causal_lm;

/// Maximum number of distinct read-only assets retained by one asset owner.
pub const MAX_RESIDENT_ASSETS: usize = crate::safe_runtime::MAX_RESIDENT_ASSETS;

/// Maximum aggregate bytes of distinct read-only assets retained by one asset owner.
pub const MAX_RESIDENT_ASSET_BYTES: u64 = crate::safe_runtime::MAX_RESIDENT_ASSET_BYTES;

/// A compiled program retaining the execution worker that created it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Program {
    artifact: Artifact,
}

impl Program {
    pub fn entry_points(&self) -> &[EntryPoint] {
        self.artifact.entry_points()
    }
}

/// Opaque identity for one read-only asset resident in an asset owner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asset {
    owner: RuntimeId,
    resident: u64,
    byte_len: u64,
}

impl Asset {
    /// Length of the verified file backing this asset.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A bounds-checked view into an owner-resident [`Asset`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetSlice {
    asset: Asset,
    offset: u64,
    byte_len: u64,
}

impl AssetSlice {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Usage of an owner's immutable weight allocations at the time of the query.
/// GPU context memory and private execution state are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetUsage {
    /// Number of distinct owned allocations; shared imports are not counted.
    pub allocation_count: u64,
    /// Successfully uploaded source payload bytes, excluding allocation padding.
    /// Reattaching a cached content key does not add uploaded bytes.
    pub uploaded_bytes: u64,
    /// Owned device bytes, including padding to GPU allocation granularity.
    pub device_bytes: u64,
}

/// A process-isolated owner of immutable GPU assets shared by execution sessions.
///
/// Keeping this owner alive lets a replacement [`Session`] reuse uploaded assets
/// after an execution worker is recycled or fails. Programs and mutable model
/// state still belong to their individual execution session.
#[derive(Debug, Clone)]
pub struct AssetOwner {
    runtime: Arc<SafeRuntime>,
}

impl AssetOwner {
    pub fn with_backend(
        backend: Backend,
        timeouts: SessionTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_asset_owner(backend, timeouts).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    pub fn new(dialect: GpuDialect) -> Result<Self, SafeInitError> {
        Self::with_backend(dialect.into(), SessionTimeouts::default())
    }

    pub fn dialect(&self) -> GpuDialect {
        self.runtime.dialect()
    }

    /// Whether the owner's worker is still available for asset operations.
    /// This is a health snapshot; a later operation can still fail.
    pub fn is_available(&self) -> bool {
        self.runtime.asset_owner_is_available()
    }

    /// Query owned weight allocation usage without counting shared imports,
    /// GPU context memory, or private execution state.
    pub fn usage(&self) -> Result<AssetUsage, AssetError> {
        let (allocation_count, uploaded_bytes, device_bytes) = self.runtime.asset_usage()?;
        Ok(AssetUsage {
            allocation_count,
            uploaded_bytes,
            device_bytes,
        })
    }

    /// Upload a verified immutable file once for all sessions using this owner.
    /// The caller is responsible for verifying the content identity in `key`.
    pub fn attach(&self, key: [u8; 32], file: File) -> Result<Asset, AssetError> {
        let attached = self.runtime.attach_asset(key, file)?;
        Ok(Asset {
            owner: self.runtime.asset_owner_id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }
}

/// One process-isolated GPU worker session.
#[derive(Debug)]
pub struct Session {
    runtime: Arc<SafeRuntime>,
}

impl Session {
    /// Start an execution worker sharing immutable assets with this owner.
    pub fn with_assets(
        owner: &AssetOwner,
        timeouts: SessionTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_shared_assets(Arc::clone(&owner.runtime), timeouts).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    /// Select a backend inside the isolated worker. Auto tries CUDA then HIP;
    /// native runtime libraries are never probed in the parent process.
    pub fn with_backend(
        backend: Backend,
        timeouts: SessionTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_backend(backend, timeouts).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    /// The dialect selected for this session's device, programs, and assets.
    pub fn dialect(&self) -> GpuDialect {
        self.runtime.dialect()
    }

    /// Start a worker for the selected provider-local GPU dialect.
    pub fn new(dialect: GpuDialect) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(dialect, SessionTimeouts::default()).map(|runtime| {
            Self {
                runtime: Arc::new(runtime),
            }
        })
    }

    /// Start a worker with a provider-selected deadline for complete Catena
    /// source loading and GPU compilation.
    pub fn with_compile_timeout(
        dialect: GpuDialect,
        compile_timeout: Duration,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(
            dialect,
            SessionTimeouts::default().with_compile_timeout(compile_timeout),
        )
        .map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    /// Start a worker with provider-selected compile and execution deadlines.
    ///
    /// The execution deadline covers each control operation and the worker-side
    /// protocol for a complete causal-LM generation. It is armed once for a
    /// generation rather than once per decoded token. Operations are serialized
    /// within a session: every caller, including model drop, may first wait for
    /// the in-flight operation holding the session, and its deadline starts only
    /// after it acquires the worker. A synchronous token callback runs on the
    /// calling thread; the watchdog remains armed but cannot preempt that code.
    pub fn with_timeouts(
        dialect: GpuDialect,
        timeouts: SessionTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(dialect, timeouts).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    /// Compile one verified program with Catena's embedded standard library.
    pub fn prepare(&mut self, source: &str) -> Result<Program, SafeInitError> {
        let mut sources: Vec<&str> = crate::stdlib::sources().collect();
        sources.push(source);
        self.runtime
            .load_sources(sources)
            .map(|artifact| Program { artifact })
    }

    /// Attach an already-open, verified file as a persistent read-only asset.
    ///
    /// `key` is opaque to Catena: the caller is responsible for verifying that
    /// the immutable file contents have this identity. Catena passes only the
    /// descriptor to its asset owner; it never receives a path or copies file
    /// bytes through the host. Sessions using the same owner can reuse the asset.
    pub fn attach(&self, key: [u8; 32], file: File) -> Result<Asset, AssetError> {
        let attached = self.runtime.attach_asset(key, file)?;
        Ok(Asset {
            owner: self.runtime.asset_owner_id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }

    /// Construct a checked view into an asset belonging to this session's owner.
    pub fn slice(
        &self,
        asset: &Asset,
        offset: u64,
        byte_len: u64,
    ) -> Result<AssetSlice, AssetError> {
        if asset.owner != self.runtime.asset_owner_id() {
            return Err(AssetError::WrongSession);
        }
        let in_bounds = offset
            .checked_add(byte_len)
            .is_some_and(|end| end <= asset.byte_len);
        if !in_bounds {
            return Err(AssetError::InvalidSlice {
                offset,
                byte_len,
                asset_byte_len: asset.byte_len,
            });
        }
        Ok(AssetSlice {
            asset: asset.clone(),
            offset,
            byte_len,
        })
    }
}

/// Run worker mode when requested by a parent [`Session`].
pub fn run_worker_if_requested() -> Result<bool, ChildMainError> {
    crate::safe_runtime::run_safe_runtime_child_if_requested()
}
