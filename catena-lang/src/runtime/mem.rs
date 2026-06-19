use std::{
    env,
    ffi::{CString, c_int, c_void},
    path::PathBuf,
    sync::Arc,
};

use libloading::Library;
use thiserror::Error;

use crate::{codegen::GpuDialect, runtime::executor::CatenaMem};

#[derive(Debug, Error)]
pub enum MemError {
    #[error(
        "failed to load {dialect:?} runtime library (tried: {tried}): {source}",
        tried = display_paths(tried)
    )]
    LoadLibrary {
        dialect: GpuDialect,
        tried: Vec<PathBuf>,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to resolve {dialect:?} runtime symbol `{symbol}`: {source}")]
    LoadSymbol {
        dialect: GpuDialect,
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("{dialect:?} runtime call failed with status {status}")]
    GpuStatus { dialect: GpuDialect, status: c_int },
}

/// Mem values represent a fat pointer (ptr + len) which can be passed into a catena program.
#[derive(Debug)]
pub struct Mem {
    pub(crate) abi: CatenaMem,
    gpu: Arc<GpuRuntime>,
}

impl Mem {
    pub fn to_u64_vec(&self) -> Vec<u64> {
        let bytes = self.abi.len as usize;
        assert_eq!(
            bytes % std::mem::size_of::<u64>(),
            0,
            "mem length is not a whole number of u64 values"
        );
        if bytes == 0 {
            return Vec::new();
        }
        let len = bytes / std::mem::size_of::<u64>();
        unsafe { std::slice::from_raw_parts(self.abi.data.cast::<u64>(), len).to_vec() }
    }

    pub(crate) fn from_u64_slice(gpu: Arc<GpuRuntime>, values: &[u64]) -> Result<Self, MemError> {
        let bytes = std::mem::size_of_val(values);
        let mut ptr = std::ptr::null_mut();
        if bytes != 0 {
            gpu.malloc_managed(&mut ptr, bytes)?;
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.cast::<u64>(), values.len());
            }
        }
        Ok(Mem {
            abi: CatenaMem {
                data: ptr,
                len: bytes as u64,
            },
            gpu,
        })
    }

    pub(crate) fn null(gpu: Arc<GpuRuntime>) -> Self {
        Self {
            abi: CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            },
            gpu,
        }
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        if !self.abi.data.is_null() {
            let _ = self.gpu.free(self.abi.data);
        }
    }
}

#[derive(Debug)]
pub(crate) struct GpuRuntime {
    dialect: GpuDialect,
    library: Library,
}

impl GpuRuntime {
    pub(crate) fn load(dialect: GpuDialect) -> Result<Self, MemError> {
        let candidates = candidate_runtime_library_paths(dialect);
        let mut last_error = None;

        for path in &candidates {
            match unsafe { Library::new(path) } {
                Ok(library) => return Ok(Self { dialect, library }),
                Err(error) => last_error = Some(error),
            }
        }

        Err(MemError::LoadLibrary {
            dialect,
            tried: candidates,
            source: last_error.expect("runtime library candidate list should not be empty"),
        })
    }

    fn malloc_managed(&self, ptr: &mut *mut c_void, bytes: usize) -> Result<(), MemError> {
        let symbol = match self.dialect {
            GpuDialect::Hip => "hipMallocManaged",
            GpuDialect::Cuda => "cudaMallocManaged",
        };
        let symbol_cstr =
            CString::new(symbol).expect("runtime symbol names should not contain NUL");
        let malloc = unsafe {
            self.library
                .get::<unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int>(
                    symbol_cstr.as_bytes_with_nul(),
                )
                .map_err(|source| MemError::LoadSymbol {
                    dialect: self.dialect,
                    symbol,
                    source,
                })?
        };
        gpu_check(self.dialect, unsafe { malloc(ptr, bytes, 1) })
    }

    fn free(&self, ptr: *mut c_void) -> Result<(), MemError> {
        let symbol = match self.dialect {
            GpuDialect::Hip => "hipFree",
            GpuDialect::Cuda => "cudaFree",
        };
        let symbol_cstr =
            CString::new(symbol).expect("runtime symbol names should not contain NUL");
        let free = unsafe {
            self.library
                .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(symbol_cstr.as_bytes_with_nul())
                .map_err(|source| MemError::LoadSymbol {
                    dialect: self.dialect,
                    symbol,
                    source,
                })?
        };
        gpu_check(self.dialect, unsafe { free(ptr) })
    }
}

fn gpu_check(dialect: GpuDialect, status: c_int) -> Result<(), MemError> {
    if status == 0 {
        Ok(())
    } else {
        Err(MemError::GpuStatus { dialect, status })
    }
}

fn candidate_runtime_library_paths(dialect: GpuDialect) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    match dialect {
        GpuDialect::Hip => {
            candidates.push(PathBuf::from("libamdhip64.so"));
            for env_var in ["ROCM_PATH", "HIP_PATH"] {
                if let Some(root) = env::var_os(env_var) {
                    candidates.push(PathBuf::from(&root).join("lib/libamdhip64.so"));
                }
            }
        }
        GpuDialect::Cuda => {
            candidates.push(PathBuf::from("libcudart.so"));
            for env_var in ["CUDA_PATH", "CUDA_HOME"] {
                if let Some(root) = env::var_os(env_var) {
                    let root = PathBuf::from(root);
                    candidates.push(root.join("lib64/libcudart.so"));
                    candidates.push(root.join("lib/libcudart.so"));
                }
            }
        }
    }
    candidates
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
