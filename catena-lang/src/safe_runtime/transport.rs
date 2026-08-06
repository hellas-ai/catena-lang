//! SafeRuntime-owned GPU allocation and IPC transport.
//!
//! This module deliberately uses only public Runtime memory types. Imported
//! mappings remain transport guards; they never become `MemOwn`.

use std::{
    env,
    ffi::{CString, c_int, c_uint, c_void},
    path::PathBuf,
    sync::Arc,
};

use libloading::{Library, Symbol};
use thiserror::Error;

use crate::{
    codegen::GpuDialect,
    runtime::{MemError, MemOwn},
};

const IPC_HANDLE_BYTES: usize = 64;
const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const IPC_LAZY_ENABLE_PEER_ACCESS: c_uint = 1;

#[derive(Debug, Error)]
pub enum SafeMemoryError {
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
    #[error("{dialect:?} runtime failed to {operation} with status {status}")]
    GpuOperation {
        dialect: GpuDialect,
        operation: &'static str,
        status: c_int,
    },
    #[error("device memory length {byte_len} cannot be represented on this platform")]
    LengthTooLarge { byte_len: u64 },
    #[error("IPC handle has {actual} bytes; expected 64")]
    InvalidIpcHandleLength { actual: usize },
    #[error("device memory uses {actual:?}, but SafeRuntime uses {expected:?}")]
    IncompatibleDialect {
        expected: GpuDialect,
        actual: GpuDialect,
    },
    #[error("non-empty device memory has a null pointer")]
    NullDevicePointer,
    #[error("Runtime returned a cap.ref memory value; cap.ref outputs are unsupported")]
    UnsupportedRefOutput,
    #[error(transparent)]
    RuntimeMemory(#[from] MemError),
}

#[derive(Debug, Clone)]
pub(crate) struct GpuTransport {
    gpu: Arc<GpuTransportApi>,
}

impl GpuTransport {
    pub(crate) fn new(dialect: GpuDialect) -> Result<Self, SafeMemoryError> {
        Ok(Self {
            gpu: Arc::new(GpuTransportApi::load(dialect)?),
        })
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }

    pub(crate) fn allocate_from_host(
        &self,
        bytes: &[u8],
    ) -> Result<OwnedAllocation, SafeMemoryError> {
        let allocation = self.allocate(bytes.len() as u64)?;
        if !bytes.is_empty() {
            self.gpu.copy(
                allocation.data,
                bytes.as_ptr().cast(),
                bytes.len(),
                MEMCPY_HOST_TO_DEVICE,
                "copy host to device",
            )?;
            self.gpu.synchronize()?;
        }
        Ok(allocation)
    }

    pub(crate) fn copy_from_device(
        &self,
        data: *const c_void,
        byte_len: u64,
        dialect: GpuDialect,
    ) -> Result<OwnedAllocation, SafeMemoryError> {
        self.check_dialect(dialect)?;
        let allocation = self.allocate(byte_len)?;
        let byte_len =
            usize::try_from(byte_len).map_err(|_| SafeMemoryError::LengthTooLarge { byte_len })?;
        if byte_len != 0 {
            if data.is_null() {
                return Err(SafeMemoryError::NullDevicePointer);
            }
            self.gpu.copy(
                allocation.data,
                data,
                byte_len,
                MEMCPY_DEVICE_TO_DEVICE,
                "copy device to device",
            )?;
            self.gpu.synchronize()?;
        }
        Ok(allocation)
    }

    pub(crate) fn export(
        &self,
        data: *mut c_void,
        byte_len: u64,
        dialect: GpuDialect,
    ) -> Result<IpcHandle, SafeMemoryError> {
        self.check_dialect(dialect)?;
        if byte_len != 0 && data.is_null() {
            return Err(SafeMemoryError::NullDevicePointer);
        }
        let mut raw = RawIpcMemHandle {
            bytes: [0; IPC_HANDLE_BYTES],
        };
        if byte_len != 0 {
            self.gpu.ipc_get(&mut raw, data)?;
        }
        Ok(IpcHandle {
            dialect,
            byte_len,
            bytes: raw.bytes,
        })
    }

    pub(crate) fn open(&self, handle: IpcHandle) -> Result<IpcMapping, SafeMemoryError> {
        self.check_dialect(handle.dialect)?;
        let byte_len =
            usize::try_from(handle.byte_len).map_err(|_| SafeMemoryError::LengthTooLarge {
                byte_len: handle.byte_len,
            })?;
        let mut data = std::ptr::null_mut();
        if byte_len != 0 {
            self.gpu.ipc_open(
                &mut data,
                RawIpcMemHandle {
                    bytes: handle.bytes,
                },
            )?;
        }
        Ok(IpcMapping {
            data,
            byte_len: handle.byte_len,
            gpu: self.gpu.clone(),
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), SafeMemoryError> {
        self.gpu.synchronize()
    }

    fn allocate(&self, byte_len: u64) -> Result<OwnedAllocation, SafeMemoryError> {
        let size =
            usize::try_from(byte_len).map_err(|_| SafeMemoryError::LengthTooLarge { byte_len })?;
        let mut data = std::ptr::null_mut();
        if size != 0 {
            self.gpu.malloc(&mut data, size)?;
        }
        Ok(OwnedAllocation {
            data,
            byte_len,
            gpu: self.gpu.clone(),
        })
    }

    fn check_dialect(&self, actual: GpuDialect) -> Result<(), SafeMemoryError> {
        if actual == self.dialect() {
            Ok(())
        } else {
            Err(SafeMemoryError::IncompatibleDialect {
                expected: self.dialect(),
                actual,
            })
        }
    }
}

#[derive(Debug)]
pub(crate) struct OwnedAllocation {
    data: *mut c_void,
    byte_len: u64,
    gpu: Arc<GpuTransportApi>,
}

impl OwnedAllocation {
    pub(crate) fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }

    pub(crate) fn export(&self) -> Result<IpcHandle, SafeMemoryError> {
        GpuTransport {
            gpu: self.gpu.clone(),
        }
        .export(self.data, self.byte_len, self.dialect())
    }

    pub(crate) fn into_mem_own(mut self) -> Result<MemOwn, SafeMemoryError> {
        // SAFETY: this is a unique ordinary allocation from the matching GPU
        // allocator. It is not an imported IPC mapping.
        let memory = unsafe { MemOwn::from_raw_parts(self.data, self.byte_len, self.dialect()) }?;
        self.data = std::ptr::null_mut();
        Ok(memory)
    }
}

impl Drop for OwnedAllocation {
    fn drop(&mut self) {
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        let _ = self.gpu.free(data);
    }
}

#[derive(Debug)]
pub(crate) struct IpcMapping {
    data: *mut c_void,
    byte_len: u64,
    gpu: Arc<GpuTransportApi>,
}

impl IpcMapping {
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.data
    }

    pub(crate) fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }
}

impl Drop for IpcMapping {
    fn drop(&mut self) {
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        let _ = self.gpu.ipc_close(data);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IpcHandle {
    dialect: GpuDialect,
    byte_len: u64,
    bytes: [u8; IPC_HANDLE_BYTES],
}

impl IpcHandle {
    pub(crate) fn from_bytes(
        dialect: GpuDialect,
        byte_len: u64,
        bytes: Vec<u8>,
    ) -> Result<Self, SafeMemoryError> {
        let actual = bytes.len();
        let bytes = bytes
            .try_into()
            .map_err(|_| SafeMemoryError::InvalidIpcHandleLength { actual })?;
        Ok(Self {
            dialect,
            byte_len,
            bytes,
        })
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub(crate) fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes.to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawIpcMemHandle {
    bytes: [u8; IPC_HANDLE_BYTES],
}

#[derive(Debug)]
struct GpuTransportApi {
    dialect: GpuDialect,
    library: Library,
}

impl GpuTransportApi {
    fn load(dialect: GpuDialect) -> Result<Self, SafeMemoryError> {
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
        Err(SafeMemoryError::LoadLibrary {
            dialect,
            tried,
            source: last_error,
        })
    }

    fn malloc(&self, data: &mut *mut c_void, byte_len: usize) -> Result<(), SafeMemoryError> {
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
    ) -> Result<(), SafeMemoryError> {
        let symbol = self.symbol("hipMemcpy", "cudaMemcpy");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check(operation, unsafe {
            function(destination, source, byte_len, kind)
        })
    }

    fn synchronize(&self) -> Result<(), SafeMemoryError> {
        let symbol = self.symbol("hipDeviceSynchronize", "cudaDeviceSynchronize");
        let function: Symbol<'_, unsafe extern "C" fn() -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("synchronize device", unsafe { function() })
    }

    fn ipc_get(
        &self,
        handle: &mut RawIpcMemHandle,
        data: *mut c_void,
    ) -> Result<(), SafeMemoryError> {
        let symbol = self.symbol("hipIpcGetMemHandle", "cudaIpcGetMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut RawIpcMemHandle, *mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("export IPC memory handle", unsafe {
            function(handle, data)
        })
    }

    fn ipc_open(
        &self,
        data: &mut *mut c_void,
        handle: RawIpcMemHandle,
    ) -> Result<(), SafeMemoryError> {
        let symbol = self.symbol("hipIpcOpenMemHandle", "cudaIpcOpenMemHandle");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut *mut c_void, RawIpcMemHandle, c_uint) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check("open IPC memory handle", unsafe {
            function(data, handle, IPC_LAZY_ENABLE_PEER_ACCESS)
        })
    }

    fn ipc_close(&self, data: *mut c_void) -> Result<(), SafeMemoryError> {
        if data.is_null() {
            return Ok(());
        }
        let symbol = self.symbol("hipIpcCloseMemHandle", "cudaIpcCloseMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("close IPC memory handle", unsafe { function(data) })
    }

    fn free(&self, data: *mut c_void) -> Result<(), SafeMemoryError> {
        if data.is_null() {
            return Ok(());
        }
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

    unsafe fn load_symbol<T>(
        &self,
        symbol: &'static str,
    ) -> Result<Symbol<'_, T>, SafeMemoryError> {
        let symbol_cstr =
            CString::new(symbol).expect("runtime symbol names should not contain NUL");
        unsafe { self.library.get::<T>(symbol_cstr.as_bytes_with_nul()) }.map_err(|source| {
            SafeMemoryError::LoadSymbol {
                dialect: self.dialect,
                symbol,
                source,
            }
        })
    }

    fn check(&self, operation: &'static str, status: c_int) -> Result<(), SafeMemoryError> {
        if status == 0 {
            Ok(())
        } else {
            Err(SafeMemoryError::GpuOperation {
                dialect: self.dialect,
                operation,
                status,
            })
        }
    }
}

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

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_handle_validates_wire_length() {
        let error = IpcHandle::from_bytes(GpuDialect::Hip, 8, vec![0; 63]).unwrap_err();
        assert!(matches!(
            error,
            SafeMemoryError::InvalidIpcHandleLength { actual: 63 }
        ));
    }

    #[test]
    fn fallback_paths_are_bounded_and_deduplicated() {
        let paths = runtime_library_fallback_paths(GpuDialect::Hip);
        assert!(paths.len() <= 2);
        assert_eq!(
            paths.iter().collect::<std::collections::HashSet<_>>().len(),
            paths.len()
        );
    }
}
