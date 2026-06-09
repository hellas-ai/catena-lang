//! Compile generated HIP/C++ code to a shared object file.

use std::{
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("failed to create temporary build directory or copy generated source: {0}")]
    TempDir(#[from] std::io::Error),
    #[error("hipcc is unavailable: {0}")]
    CompilerUnavailable(std::io::Error),
    #[error("HIP/C++ compilation failed with status {status}: {stderr}")]
    CompilerFailed { status: ExitStatus, stderr: String },
}

/// A shared object file created by compiling generated Catena HIP/C++.
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

pub(crate) fn compile(cpp_path: &Path) -> Result<Artifact, ArtifactError> {
    let build_dir = tempfile::Builder::new().prefix("catena-hip-").tempdir()?;
    let module_path = build_dir.path().join("module.cpp");
    let so_path = build_dir.path().join("module.so");
    std::fs::copy(cpp_path, &module_path)?;

    let output = Command::new("hipcc")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O2")
        .arg("--std=c++17")
        // Keep multiply/add as separately rounded operations. This prevents a
        // future generated `a * b + c` expression from being contracted to FMA.
        .arg("-ffp-contract=off")
        // This is the default, but keep it explicit because reproducibility
        // depends on avoiding reassociation and other fast-math transforms.
        .arg("-fno-fast-math")
        .arg(&module_path)
        .arg("-o")
        .arg(&so_path)
        .output()
        .map_err(ArtifactError::CompilerUnavailable)?;

    if !output.status.success() {
        return Err(ArtifactError::CompilerFailed {
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    Ok(Artifact {
        _build_dir: build_dir,
        path: so_path,
    })
}
