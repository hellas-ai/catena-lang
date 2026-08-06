use std::{
    env,
    ffi::{CString, c_int, c_void},
    path::PathBuf,
};

use libloading::{Library, Symbol};

use super::mem::MemError;
use crate::codegen::GpuDialect;

const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// The process-local HIP/CUDA operations needed to manage a [`super::MemOwn`].
///
/// Transport-specific operations, including IPC and device-to-device copies,
/// deliberately live outside Runtime.
#[derive(Debug)]
pub(super) struct GpuApi {
    dialect: GpuDialect,
    library: Library,
}

impl GpuApi {
    pub(super) fn load(dialect: GpuDialect) -> Result<Self, MemError> {
        let library_name = match dialect {
            GpuDialect::Hip => "libamdhip64.so",
            GpuDialect::Cuda => "libcudart.so",
        };
        let mut tried = vec![PathBuf::from(library_name)];
        let mut last_error = match unsafe { Library::new(library_name) } {
            Ok(library) => return Ok(Self { dialect, library }),
            Err(error) => error,
        };

        for path in runtime_library_fallback_paths(dialect) {
            tried.push(path.clone());
            match unsafe { Library::new(path) } {
                Ok(library) => return Ok(Self { dialect, library }),
                Err(error) => last_error = error,
            }
        }

        Err(MemError::LoadLibrary {
            dialect,
            tried,
            source: last_error,
        })
    }

    pub(super) fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub(super) fn allocate(&self, byte_len: usize) -> Result<*mut c_void, MemError> {
        if byte_len == 0 {
            return Ok(std::ptr::null_mut());
        }

        let mut data = std::ptr::null_mut();
        let symbol = self.symbol("hipMalloc", "cudaMalloc");
        let function: Symbol<'_, unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("allocate device memory", unsafe {
            function(&mut data, byte_len)
        })?;
        Ok(data)
    }

    pub(super) fn copy_host_to_device(
        &self,
        destination: *mut c_void,
        source: &[u8],
    ) -> Result<(), MemError> {
        if source.is_empty() {
            return Ok(());
        }
        self.copy(
            destination,
            source.as_ptr().cast(),
            source.len(),
            MEMCPY_HOST_TO_DEVICE,
            "copy host to device",
        )
    }

    pub(super) fn copy_device_to_host(
        &self,
        source: *const c_void,
        output: &mut [u8],
    ) -> Result<(), MemError> {
        if output.is_empty() {
            return Ok(());
        }
        self.copy(
            output.as_mut_ptr().cast(),
            source,
            output.len(),
            MEMCPY_DEVICE_TO_HOST,
            "copy device to host",
        )
    }

    pub(super) fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        if data.is_null() {
            return Ok(());
        }
        let symbol = self.symbol("hipFree", "cudaFree");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("free device memory", unsafe { function(data) })
    }

    fn copy(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        byte_len: usize,
        kind: c_int,
        operation: &'static str,
    ) -> Result<(), MemError> {
        let symbol = self.symbol("hipMemcpy", "cudaMemcpy");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check(operation, unsafe {
            function(destination, source, byte_len, kind)
        })
    }

    fn symbol(&self, hip: &'static str, cuda: &'static str) -> &'static str {
        match self.dialect {
            GpuDialect::Hip => hip,
            GpuDialect::Cuda => cuda,
        }
    }

    unsafe fn load_symbol<T>(&self, symbol: &'static str) -> Result<Symbol<'_, T>, MemError> {
        let symbol_cstr =
            CString::new(symbol).expect("runtime symbol names should not contain NUL");
        unsafe { self.library.get::<T>(symbol_cstr.as_bytes_with_nul()) }.map_err(|source| {
            MemError::LoadSymbol {
                dialect: self.dialect,
                symbol,
                source,
            }
        })
    }

    fn check(&self, operation: &'static str, status: c_int) -> Result<(), MemError> {
        if status == 0 {
            Ok(())
        } else {
            Err(MemError::GpuOperation {
                dialect: self.dialect,
                operation,
                status,
            })
        }
    }
}

/// Return a small, ordered set of runtime-library locations.
///
/// The bare soname delegates discovery to the platform loader. The remaining
/// paths come only from toolkit roots explicitly configured by the user or
/// development environment; this intentionally performs no filesystem search.
fn runtime_library_fallback_paths(dialect: GpuDialect) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    match dialect {
        GpuDialect::Hip => {
            for variable in ["ROCM_PATH", "HIP_PATH"] {
                if let Some(root) = env::var_os(variable) {
                    push_unique(&mut paths, PathBuf::from(root).join("lib/libamdhip64.so"));
                }
            }
        }
        GpuDialect::Cuda => {
            for variable in ["CUDA_PATH", "CUDA_HOME"] {
                if let Some(root) = env::var_os(variable) {
                    let root = PathBuf::from(root);
                    push_unique(&mut paths, root.join("lib64/libcudart.so"));
                    push_unique(&mut paths, root.join("lib/libcudart.so"));
                }
            }
        }
    }
    paths
}

fn push_unique(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}
