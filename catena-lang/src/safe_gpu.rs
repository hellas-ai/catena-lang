//! Session-oriented, process-isolated GPU preparation.
//!
//! [`Session`] owns one long-lived worker and every [`Program`] loaded into it.
//! A loaded program exposes its immutable entry-point metadata so an adapter can
//! reject an incompatible dynamic ABI before execution.
//!
//! Raw value execution is intentionally not exposed here. The typed
//! [`causal_lm`] interface keeps read-only assets and mutable model state in the
//! worker instead of routing model memory through a per-call transport. The
//! lower-level [`crate::safe_runtime`] remains available for existing
//! transient-value users.

use std::{fs::File, sync::Arc};

use crate::{
    runtime::{Artifact, RuntimeId},
    safe_runtime::SafeRuntime,
};

pub use crate::{
    codegen::GpuDialect,
    runtime::{EntryPoint, ValueKind},
    safe_runtime::{AssetError, ChildMainError, SafeInitError},
};

pub mod causal_lm;

/// A program compiled and retained by a [`Session`].
pub type Program = Artifact;

/// Opaque identity for one read-only asset resident in a GPU session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asset {
    session: RuntimeId,
    resident: u64,
    byte_len: u64,
}

impl Asset {
    /// Length of the verified file backing this asset.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A bounds-checked view into a session-resident [`Asset`].
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

/// One process-isolated GPU worker session.
#[derive(Debug)]
pub struct Session {
    runtime: Arc<SafeRuntime>,
}

impl Session {
    /// Start a worker for the selected provider-local GPU dialect.
    pub fn new(dialect: GpuDialect) -> Result<Self, SafeInitError> {
        SafeRuntime::new(dialect).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    /// Compile one verified program with Catena's embedded standard library.
    pub fn prepare(&mut self, source: &str) -> Result<Program, SafeInitError> {
        let mut sources: Vec<&str> = crate::stdlib::sources().collect();
        sources.push(source);
        self.runtime.load_sources(sources)
    }

    /// Attach an already-open, verified file as a persistent read-only asset.
    ///
    /// `key` is opaque to Catena: the caller is responsible for verifying that
    /// the immutable file contents have this identity. Catena passes only the
    /// descriptor to its worker; it never receives a path or copies file bytes.
    pub fn attach(&self, key: [u8; 32], file: File) -> Result<Asset, AssetError> {
        let attached = self.runtime.attach_asset(key, file)?;
        Ok(Asset {
            session: self.runtime.id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }

    /// Construct a checked view into an asset belonging to this session.
    pub fn slice(
        &self,
        asset: &Asset,
        offset: u64,
        byte_len: u64,
    ) -> Result<AssetSlice, AssetError> {
        if asset.session != self.runtime.id() {
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
