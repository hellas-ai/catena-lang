use std::{
    ffi::{c_int, c_void},
    path::{Path, PathBuf},
    sync::Arc,
};

use libloading::Library;
use thiserror::Error;

use crate::runtime::executor::CatenaMem;

#[derive(Debug, Error)]
pub enum MemError {
    #[error("failed to load HIP runtime library: {0}")]
    LoadLibrary(#[source] libloading::Error),
    #[error("failed to resolve HIP runtime symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("HIP call failed with status {0}")]
    HipStatus(c_int),
}

/// Mem values represent a fat pointer (ptr + len) which can be passed into a catena program.
#[derive(Debug)]
pub struct Mem {
    pub(crate) abi: CatenaMem,
    hip: Arc<Hip>,
}

impl Mem {
    pub(crate) fn from_u64_slice(hip: Arc<Hip>, values: &[u64]) -> Result<Self, MemError> {
        let bytes = std::mem::size_of_val(values);
        let mut ptr = std::ptr::null_mut();
        if bytes != 0 {
            hip.malloc_managed(&mut ptr, bytes)?;
            unsafe {
                std::ptr::copy_nonoverlapping(values.as_ptr(), ptr.cast::<u64>(), values.len());
            }
        }
        Ok(Mem {
            abi: CatenaMem {
                data: ptr,
                len: bytes as u64,
            },
            hip,
        })
    }

    pub(crate) fn null(hip: Arc<Hip>) -> Self {
        Self {
            abi: CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            },
            hip,
        }
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        if !self.abi.data.is_null() {
            let _ = self.hip.free(self.abi.data);
        }
    }
}

#[derive(Debug)]
pub(crate) struct Hip {
    library: Library,
}

impl Hip {
    pub(crate) fn load() -> Result<Self, MemError> {
        if let Ok(library) = unsafe { Library::new("libamdhip64.so") } {
            return Ok(Self { library });
        }
        for env_var in ["ROCM_PATH", "HIP_PATH"] {
            let Ok(root) = std::env::var(env_var) else {
                continue;
            };
            let path = PathBuf::from(root).join("lib/libamdhip64.so");
            if let Ok(library) = unsafe { Library::new(path) } {
                return Ok(Self { library });
            }
        }
        unsafe { Library::new(Path::new("libamdhip64.so")) }
            .map(|library| Self { library })
            .map_err(MemError::LoadLibrary)
    }

    fn malloc_managed(&self, ptr: &mut *mut c_void, bytes: usize) -> Result<(), MemError> {
        let malloc = unsafe {
            self.library
                .get::<unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int>(
                    b"hipMallocManaged\0",
                )
                .map_err(|source| MemError::LoadSymbol {
                    symbol: "hipMallocManaged",
                    source,
                })?
        };
        hip_check(unsafe { malloc(ptr, bytes, 1) })
    }

    fn free(&self, ptr: *mut c_void) -> Result<(), MemError> {
        let free = unsafe {
            self.library
                .get::<unsafe extern "C" fn(*mut c_void) -> c_int>(b"hipFree\0")
                .map_err(|source| MemError::LoadSymbol {
                    symbol: "hipFree",
                    source,
                })?
        };
        hip_check(unsafe { free(ptr) })
    }
}

fn hip_check(status: c_int) -> Result<(), MemError> {
    if status == 0 {
        Ok(())
    } else {
        Err(MemError::HipStatus(status))
    }
}
