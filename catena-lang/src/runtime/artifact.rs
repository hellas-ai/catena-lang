//! Compile generated GPU C++ code to a shared object file.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
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

/// A shared object file created by compiling generated Catena GPU C++.
#[derive(Debug)]
pub(crate) struct Artifact {
    _build_dir: tempfile::TempDir,
    path: PathBuf,
}

impl Artifact {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

pub(crate) fn compile(cpp_path: &Path, dialect: GpuDialect) -> Result<Artifact, ArtifactError> {
    let build_dir = tempfile::Builder::new()
        .prefix("catena-module-")
        .tempdir()?;
    let module_path = build_dir.path().join("module.cpp");
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

    Ok(Artifact {
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
