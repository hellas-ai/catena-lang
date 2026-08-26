use std::{
    env,
    ffi::{CString, c_int, c_void},
    path::PathBuf,
    sync::Arc,
};

use libloading::{Library, Symbol};

use crate::{codegen::GpuDialect, runtime::MemError};

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

#[derive(Debug)]
pub(crate) struct GpuApi {
    dialect: GpuDialect,
    library: Library,
}

impl GpuApi {
    pub(crate) fn load(dialect: GpuDialect) -> Result<Arc<Self>, MemError> {
        let library_name = match dialect {
            GpuDialect::Hip => "libamdhip64.so",
            GpuDialect::Cuda => "libcudart.so",
        };
        let mut paths = vec![PathBuf::from(library_name)];
        let variables = match dialect {
            GpuDialect::Hip => ["ROCM_PATH", "HIP_PATH"],
            GpuDialect::Cuda => ["CUDA_PATH", "CUDA_HOME"],
        };
        for variable in variables {
            if let Some(root) = env::var_os(variable) {
                let root = PathBuf::from(root);
                match dialect {
                    GpuDialect::Hip => paths.push(root.join("lib/libamdhip64.so")),
                    GpuDialect::Cuda => {
                        paths.push(root.join("lib64/libcudart.so"));
                        paths.push(root.join("lib/libcudart.so"));
                    }
                }
            }
        }

        let mut last_error = None;
        for path in &paths {
            match unsafe { Library::new(path) } {
                Ok(library) => return Ok(Arc::new(Self { dialect, library })),
                Err(error) => last_error = Some(error),
            }
        }
        Err(MemError::LoadLibrary {
            dialect,
            paths,
            source: last_error.expect("GPU library path list is non-empty"),
        })
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub(crate) fn allocate(&self, byte_len: usize) -> Result<*mut c_void, MemError> {
        if byte_len == 0 {
            return Ok(std::ptr::null_mut());
        }
        let mut data = std::ptr::null_mut();
        let function: Symbol<'_, unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int> =
            unsafe { self.load_symbol("hipMalloc", "cudaMalloc")? };
        self.check("allocate device memory", unsafe {
            function(&mut data, byte_len)
        })?;
        Ok(data)
    }

    pub(crate) fn copy_to_device(
        &self,
        destination: *mut c_void,
        source: &[u8],
    ) -> Result<(), MemError> {
        self.copy(
            destination,
            source.as_ptr().cast(),
            source.len(),
            HOST_TO_DEVICE,
            "copy host memory to the device",
        )
    }

    pub(crate) fn copy_to_host(
        &self,
        source: *const c_void,
        destination: &mut [u8],
    ) -> Result<(), MemError> {
        self.copy(
            destination.as_mut_ptr().cast(),
            source,
            destination.len(),
            DEVICE_TO_HOST,
            "copy device memory to the host",
        )
    }

    pub(crate) fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        if data.is_null() {
            return Ok(());
        }
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol("hipFree", "cudaFree")? };
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
        if byte_len == 0 {
            return Ok(());
        }
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        > = unsafe { self.load_symbol("hipMemcpy", "cudaMemcpy")? };
        self.check(operation, unsafe {
            function(destination, source, byte_len, kind)
        })
    }

    unsafe fn load_symbol<T>(
        &self,
        hip: &'static str,
        cuda: &'static str,
    ) -> Result<Symbol<'_, T>, MemError> {
        let symbol = match self.dialect {
            GpuDialect::Hip => hip,
            GpuDialect::Cuda => cuda,
        };
        let name = CString::new(symbol).expect("GPU symbol names contain no NUL bytes");
        unsafe { self.library.get(name.as_bytes_with_nul()) }.map_err(|source| {
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
