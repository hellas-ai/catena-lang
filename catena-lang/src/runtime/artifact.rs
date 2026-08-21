//! Compile generated GPU C++ code to a shared object file.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    sync::atomic::{AtomicU64, Ordering},
};

use thiserror::Error;

use crate::codegen::GpuDialect;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("failed to create temporary build directory or copy generated source: {0}")]
    TempDir(#[from] std::io::Error),
    #[error("GPU compiler `{compiler}` is unavailable: {source}")]
    CompilerUnavailable {
        compiler: String,
        #[source]
        source: std::io::Error,
    },
    #[error("GPU compilation with `{compiler}` failed with status {status}: {stderr}")]
    CompilerFailed {
        compiler: String,
        status: ExitStatus,
        stderr: String,
    },
}

/// Identifies one compiled Catena artifact belonging to a runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Artifact {
    runtime_id: RuntimeId,
    index: usize,
}

impl Artifact {
    pub(crate) fn new(runtime_id: RuntimeId, index: usize) -> Self {
        Self { runtime_id, index }
    }

    pub(crate) fn belongs_to(&self, runtime_id: RuntimeId) -> bool {
        self.runtime_id == runtime_id
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeId(u64);

impl RuntimeId {
    pub(crate) fn new() -> Self {
        let id = NEXT_RUNTIME_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("runtime ID space exhausted");
        Self(id)
    }
}

/// A shared object file created by compiling generated Catena GPU C++.
#[derive(Debug)]
pub(super) struct SharedObject {
    _build_dir: tempfile::TempDir,
    path: PathBuf,
}

impl SharedObject {
    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

pub(super) fn compile(cpp_path: &Path, dialect: GpuDialect) -> Result<SharedObject, ArtifactError> {
    let build_dir = tempfile::Builder::new()
        .prefix("catena-module-")
        .tempdir()?;
    let module_filename = match dialect {
        GpuDialect::Hip => "module.cpp",
        GpuDialect::Cuda => "module.cu",
    };
    let module_path = build_dir.path().join(module_filename);
    let so_path = build_dir.path().join("module.so");
    std::fs::copy(cpp_path, &module_path)?;

    let compiler = gpu_compiler(dialect);
    let compiler_display = compiler.to_string_lossy().into_owned();
    let mut command = Command::new(&compiler);
    command.arg("-shared").arg("-O2");
    match dialect {
        GpuDialect::Hip => {
            command
                .arg("-fPIC")
                .arg("--std=c++17")
                // Keep multiply/add as separately rounded operations. This prevents a
                // future generated `a * b + c` expression from being contracted to FMA.
                .arg("-ffp-contract=off")
                // This is the default, but keep it explicit because reproducibility
                // depends on avoiding reassociation and other fast-math transforms.
                .arg("-fno-fast-math");
        }
        GpuDialect::Cuda => {
            command
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg("--std=c++17")
                // Default to SM_80 (Ampere and later)
                .arg("-arch=sm_80")
                // Match the no-FMA intent for generated arithmetic.
                .arg("--fmad=false");
        }
    }
    let output = command
        .arg(&module_path)
        .arg("-o")
        .arg(&so_path)
        .output()
        .map_err(|source| ArtifactError::CompilerUnavailable {
            compiler: compiler_display.clone(),
            source,
        })?;

    if !output.status.success() {
        return Err(ArtifactError::CompilerFailed {
            compiler: compiler_display,
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    Ok(SharedObject {
        _build_dir: build_dir,
        path: so_path,
    })
}

fn gpu_compiler(dialect: GpuDialect) -> OsString {
    match dialect {
        GpuDialect::Hip => OsString::from("hipcc"),
        GpuDialect::Cuda => OsString::from("nvcc"),
    }
}
