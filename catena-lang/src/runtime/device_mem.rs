//! Application-owned device memory.

use std::{
    env,
    ffi::{CString, c_int, c_void},
    path::PathBuf,
    sync::Arc,
};

use libloading::{Library, Symbol};

use super::mem::{MemError, MemOwn};
use crate::codegen::GpuDialect;

const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Allocates application-owned GPU memory.
#[derive(Debug, Clone)]
pub struct DeviceAllocator {
    gpu: Arc<DeviceGpuRuntime>,
}

impl DeviceAllocator {
    pub fn new(dialect: GpuDialect) -> Result<Self, MemError> {
        let gpu = Arc::new(DeviceGpuRuntime::load(dialect)?);
        Ok(Self { gpu })
    }

    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }

    /// Allocate uninitialized device-only memory.
    pub fn allocate(&self, byte_len: usize) -> Result<DeviceBuffer, MemError> {
        let mut data = std::ptr::null_mut();
        if byte_len != 0 {
            self.gpu.malloc(&mut data, byte_len)?;
        }
        Ok(DeviceBuffer {
            data,
            byte_len,
            gpu: self.gpu.clone(),
        })
    }

    /// Allocate device-only memory and synchronously upload all bytes.
    pub fn allocate_from_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer, MemError> {
        let mut buffer = self.allocate(bytes.len())?;
        if let Err(error) = buffer.write(0, bytes) {
            let _ = buffer.release_now();
            return Err(error);
        }
        Ok(buffer)
    }

    pub(crate) fn adopt_owned(&self, data: *mut c_void, byte_len: usize) -> DeviceBuffer {
        DeviceBuffer {
            data,
            byte_len,
            gpu: self.gpu.clone(),
        }
    }
}

/// An opaque application-owned handle to a device allocation.
#[derive(Debug)]
pub struct DeviceBuffer {
    data: *mut c_void,
    byte_len: usize,
    gpu: Arc<DeviceGpuRuntime>,
}

impl DeviceBuffer {
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }

    /// Synchronously upload bytes into a checked subrange of this allocation.
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), MemError> {
        validate_range(self.byte_len, offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let destination = unsafe { self.data.cast::<u8>().add(offset).cast::<c_void>() };
        self.gpu.copy(
            destination,
            bytes.as_ptr().cast(),
            bytes.len(),
            MEMCPY_HOST_TO_DEVICE,
            "copy host to device",
        )
    }

    /// Synchronously read a checked subrange into host memory.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), MemError> {
        validate_range(self.byte_len, offset, output.len())?;
        if output.is_empty() {
            return Ok(());
        }
        let source = unsafe { self.data.cast::<u8>().add(offset).cast::<c_void>() };
        self.gpu.copy(
            output.as_mut_ptr().cast(),
            source,
            output.len(),
            MEMCPY_DEVICE_TO_HOST,
            "copy device to host",
        )
    }

    /// Release the allocation and report release failures.
    pub fn free(mut self) -> Result<(), MemError> {
        self.release_now()
    }

    /// Transfer this allocation into an owned runtime memory value.
    pub fn into_mem_own(self) -> MemOwn {
        MemOwn::from_device_buffer(self)
    }

    pub(crate) fn data(&self) -> *mut c_void {
        self.data
    }

    pub(crate) fn read_all(&self, output: &mut [u8]) -> Result<(), MemError> {
        self.read(0, output)
    }

    /// Relinquish this allocation to generated code without freeing it.
    pub(crate) fn into_raw(mut self) {
        self.data = std::ptr::null_mut();
    }

    fn release_now(&mut self) -> Result<(), MemError> {
        if self.data.is_null() {
            return Ok(());
        }
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        self.gpu.free(data)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = self.release_now();
    }
}

fn validate_range(byte_len: usize, offset: usize, length: usize) -> Result<(), MemError> {
    let end = offset
        .checked_add(length)
        .ok_or(MemError::RangeOverflow { offset, length })?;
    if end > byte_len {
        return Err(MemError::OutOfBounds {
            offset,
            end,
            byte_len,
        });
    }
    Ok(())
}

#[derive(Debug)]
struct DeviceGpuRuntime {
    dialect: GpuDialect,
    library: Library,
}

impl DeviceGpuRuntime {
    fn load(dialect: GpuDialect) -> Result<Self, MemError> {
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

    fn malloc(&self, data: &mut *mut c_void, byte_len: usize) -> Result<(), MemError> {
        let symbol = self.symbol("hipMalloc", "cudaMalloc");
        let function: Symbol<'_, unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("allocate device memory", unsafe {
            function(data, byte_len)
        })
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

    fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        let symbol = self.symbol("hipFree", "cudaFree");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("free device memory", unsafe { function(data) })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_copy_ranges() {
        assert!(validate_range(8, 2, 6).is_ok());
        assert!(matches!(
            validate_range(8, 3, 6),
            Err(MemError::OutOfBounds {
                offset: 3,
                end: 9,
                byte_len: 8
            })
        ));
        assert!(matches!(
            validate_range(8, usize::MAX, 2),
            Err(MemError::RangeOverflow {
                offset: usize::MAX,
                length: 2
            })
        ));
    }
}
