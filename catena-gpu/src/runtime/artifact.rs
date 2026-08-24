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
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("GPU compiler `{compiler}` is unavailable: {source}")]
    CompilerUnavailable {
        compiler: String,
        source: std::io::Error,
    },
    #[error("GPU compilation with `{compiler}` failed with status {status}: {stderr}")]
    CompilerFailed {
        compiler: String,
        status: ExitStatus,
        stderr: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Artifact {
    pub(crate) runtime: RuntimeId,
    pub(crate) index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeId(u64);

impl RuntimeId {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub(crate) struct SharedObject {
    _directory: tempfile::TempDir,
    path: PathBuf,
}
impl SharedObject {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub(crate) fn compile(source: &Path, dialect: GpuDialect) -> Result<SharedObject, ArtifactError> {
    let directory = tempfile::Builder::new().prefix("catena-gpu-").tempdir()?;
    let extension = match dialect {
        GpuDialect::Hip => "cpp",
        GpuDialect::Cuda => "cu",
    };
    let module = directory.path().join(format!("module.{extension}"));
    let output = directory.path().join("module.so");
    std::fs::copy(source, &module)?;
    let compiler = match dialect {
        GpuDialect::Hip => OsString::from("hipcc"),
        GpuDialect::Cuda => OsString::from("nvcc"),
    };
    let display = compiler.to_string_lossy().into_owned();
    let mut command = Command::new(&compiler);
    command.arg("-shared").arg("-O2");
    match dialect {
        GpuDialect::Hip => {
            command.args([
                "-fPIC",
                "--std=c++17",
                "-ffp-contract=off",
                "-fno-fast-math",
            ]);
        }
        GpuDialect::Cuda => {
            command.args([
                "-Xcompiler",
                "-fPIC",
                "--std=c++17",
                "-arch=sm_80",
                "--fmad=false",
            ]);
        }
    }
    let result = command
        .arg(&module)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|source| ArtifactError::CompilerUnavailable {
            compiler: display.clone(),
            source,
        })?;
    if !result.status.success() {
        return Err(ArtifactError::CompilerFailed {
            compiler: display,
            status: result.status,
            stderr: String::from_utf8_lossy(&result.stderr).trim().into(),
        });
    }
    Ok(SharedObject {
        _directory: directory,
        path: output,
    })
}
