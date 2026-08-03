//! Device memory ownership and legacy HIP/CUDA IPC handles.
//!
//! Buffers can be allocated by the application or adopted from generated Catena
//! programs. An exported allocation remains owned by the exporting process;
//! importing a handle creates a mapping of the same VRAM allocation without
//! copying its contents.

use std::{
    env,
    ffi::{CString, c_int, c_uint, c_void},
    path::PathBuf,
    rc::Rc,
    sync::Arc,
};

use libloading::{Library, Symbol};

use super::mem::{Mem, MemError};
use crate::codegen::GpuDialect;

const IPC_HANDLE_BYTES: usize = 64;
const MEMCPY_HOST_TO_DEVICE: c_int = 1;
const MEMCPY_DEVICE_TO_HOST: c_int = 2;
const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
const IPC_LAZY_ENABLE_PEER_ACCESS: c_uint = 1;

/// An opaque legacy HIP/CUDA IPC handle for one device allocation.
///
/// The exporting process and its [`DeviceBuffer`] must remain alive until all
/// importing processes have closed their mappings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IpcMemoryHandle {
    dialect: GpuDialect,
    byte_len: u64,
    bytes: [u8; IPC_HANDLE_BYTES],
}

impl IpcMemoryHandle {
    /// Reconstruct a transported handle from its metadata and opaque bytes.
    ///
    /// The metadata must come from the same call to [`DeviceBuffer::export_ipc`]
    /// as the opaque bytes.
    pub fn from_bytes(dialect: GpuDialect, byte_len: u64, bytes: [u8; IPC_HANDLE_BYTES]) -> Self {
        Self {
            dialect,
            byte_len,
            bytes,
        }
    }

    pub fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn as_bytes(&self) -> &[u8; IPC_HANDLE_BYTES] {
        &self.bytes
    }

    pub fn into_bytes(self) -> [u8; IPC_HANDLE_BYTES] {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }
}

/// Allocates application-owned memory on the currently selected GPU device.
#[derive(Debug, Clone)]
pub struct DeviceAllocator {
    gpu: Arc<DeviceGpuRuntime>,
}

impl DeviceAllocator {
    /// Bind an allocator to the process's currently selected GPU device.
    pub fn new(dialect: GpuDialect) -> Result<Self, MemError> {
        let gpu = Arc::new(DeviceGpuRuntime::load(dialect)?);
        Ok(Self { gpu })
    }

    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect
    }

    pub(crate) fn synchronize(&self) -> Result<(), MemError> {
        self.gpu.synchronize()
    }

    /// Allocate uninitialized device-only memory.
    pub fn allocate(&self, byte_len: usize) -> Result<DeviceBuffer, MemError> {
        let mut data = std::ptr::null_mut();
        if byte_len != 0 {
            self.gpu.malloc(&mut data, byte_len)?;
        }
        Ok(self.adopt_owned(data, byte_len))
    }

    /// Allocate device-only memory and synchronously upload all bytes.
    pub fn allocate_from_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer, MemError> {
        let buffer = self.allocate(bytes.len())?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// Map an allocation exported by another process without copying it.
    ///
    /// The exporting allocation must outlive the returned mapping. The handle's
    /// dialect must match this allocator.
    pub fn import_ipc(&self, handle: &IpcMemoryHandle) -> Result<DeviceBuffer, MemError> {
        if self.dialect() != handle.dialect {
            return Err(MemError::DialectMismatch {
                allocator_dialect: self.dialect(),
                handle_dialect: handle.dialect,
            });
        }
        let byte_len = usize::try_from(handle.byte_len).map_err(|_| MemError::LengthTooLarge {
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
        Ok(DeviceBuffer::new(DeviceAllocation {
            data,
            byte_len,
            gpu: self.gpu.clone(),
            release: Release::IpcClose,
        }))
    }

    pub(crate) fn adopt_owned(&self, data: *mut c_void, byte_len: usize) -> DeviceBuffer {
        DeviceBuffer::new(DeviceAllocation {
            data,
            byte_len,
            gpu: self.gpu.clone(),
            release: Release::Free,
        })
    }
}

/// An opaque application-owned handle to a device allocation or IPC mapping.
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    allocation: Rc<DeviceAllocation>,
}

#[derive(Debug)]
struct DeviceAllocation {
    data: *mut c_void,
    byte_len: usize,
    gpu: Arc<DeviceGpuRuntime>,
    release: Release,
}

impl DeviceBuffer {
    fn new(allocation: DeviceAllocation) -> Self {
        Self {
            allocation: Rc::new(allocation),
        }
    }

    pub fn byte_len(&self) -> usize {
        self.allocation.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.allocation.byte_len == 0
    }

    pub fn dialect(&self) -> GpuDialect {
        self.allocation.gpu.dialect
    }

    /// Synchronously upload bytes into a checked subrange of this allocation.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), MemError> {
        validate_range(self.byte_len(), offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let destination = unsafe { self.data().cast::<u8>().add(offset).cast::<c_void>() };
        self.allocation.gpu.copy(
            destination,
            bytes.as_ptr().cast(),
            bytes.len(),
            MEMCPY_HOST_TO_DEVICE,
            "copy host to device",
        )
    }

    /// Synchronously read a checked subrange into host memory.
    pub fn read(&self, offset: usize, output: &mut [u8]) -> Result<(), MemError> {
        validate_range(self.byte_len(), offset, output.len())?;
        if output.is_empty() {
            return Ok(());
        }
        let source = unsafe { self.data().cast::<u8>().add(offset).cast::<c_void>() };
        self.allocation.gpu.copy(
            output.as_mut_ptr().cast(),
            source,
            output.len(),
            MEMCPY_DEVICE_TO_HOST,
            "copy device to host",
        )
    }

    /// Synchronously copy between checked subranges of two device allocations.
    pub(crate) fn copy_from_device(
        &self,
        destination_offset: usize,
        source: &Self,
        source_offset: usize,
        byte_len: usize,
    ) -> Result<(), MemError> {
        validate_range(self.byte_len(), destination_offset, byte_len)?;
        validate_range(source.byte_len(), source_offset, byte_len)?;
        if byte_len == 0 {
            return Ok(());
        }
        let destination = unsafe {
            self.data()
                .cast::<u8>()
                .add(destination_offset)
                .cast::<c_void>()
        };
        let source = unsafe {
            source
                .data()
                .cast::<u8>()
                .add(source_offset)
                .cast::<c_void>()
        };
        self.allocation.gpu.copy(
            destination,
            source,
            byte_len,
            MEMCPY_DEVICE_TO_DEVICE,
            "copy device to device",
        )?;
        self.allocation.gpu.synchronize()
    }

    /// Export an opaque handle that another process can import.
    ///
    /// Imported mappings cannot be exported again.
    pub fn export_ipc(&self) -> Result<IpcMemoryHandle, MemError> {
        if self.allocation.release == Release::IpcClose {
            return Err(MemError::CannotExportImported);
        }
        let mut raw = RawIpcMemHandle {
            bytes: [0; IPC_HANDLE_BYTES],
        };
        if !self.data().is_null() {
            self.allocation.gpu.ipc_get(&mut raw, self.data())?;
        }
        Ok(IpcMemoryHandle {
            dialect: self.allocation.gpu.dialect,
            byte_len: self.allocation.byte_len as u64,
            bytes: raw.bytes,
        })
    }

    /// Release a uniquely owned allocation or mapping and report failures.
    /// Returns [`MemError::AllocationShared`] while clones still exist.
    pub fn free(self) -> Result<(), MemError> {
        let allocation = Rc::try_unwrap(self.allocation).map_err(|_| MemError::AllocationShared)?;
        allocation.release_now()
    }

    /// Transfer this allocation or imported mapping into a runtime memory value.
    pub fn into_mem(self) -> Mem {
        Mem::from_device_buffer(self)
    }

    pub(crate) fn data(&self) -> *mut c_void {
        self.allocation.data
    }

    pub(crate) fn into_mem_view(self, offset: u64, byte_len: u64) -> Result<Mem, MemError> {
        Mem::from_device_buffer_view(self, offset, byte_len)
    }

    pub(crate) fn data_for_view(
        &self,
        offset: u64,
        byte_len: u64,
    ) -> Result<*mut c_void, MemError> {
        let offset =
            usize::try_from(offset).map_err(|_| MemError::LengthTooLarge { byte_len: offset })?;
        let byte_len =
            usize::try_from(byte_len).map_err(|_| MemError::LengthTooLarge { byte_len })?;
        validate_range(self.byte_len(), offset, byte_len)?;
        if offset == 0 {
            return Ok(self.data());
        }
        Ok(unsafe { self.data().cast::<u8>().add(offset).cast() })
    }

    pub(crate) fn view_offset(&self, data: *mut c_void, byte_len: u64) -> Result<u64, MemError> {
        let offset = (data as usize)
            .checked_sub(self.data() as usize)
            .ok_or(MemError::InvalidView)?;
        let byte_len =
            usize::try_from(byte_len).map_err(|_| MemError::LengthTooLarge { byte_len })?;
        validate_range(self.byte_len(), offset, byte_len)?;
        Ok(offset as u64)
    }

    pub(crate) fn same_allocation(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.allocation, &other.allocation)
    }
}

impl DeviceAllocation {
    fn release_now(mut self) -> Result<(), MemError> {
        if self.data.is_null() {
            return Ok(());
        }
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        self.gpu.release(data, self.release)
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if self.data.is_null() {
            return;
        }
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        let _ = self.gpu.release(data, self.release);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Release {
    Free,
    IpcClose,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawIpcMemHandle {
    bytes: [u8; IPC_HANDLE_BYTES],
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

    fn synchronize(&self) -> Result<(), MemError> {
        let symbol = self.symbol("hipDeviceSynchronize", "cudaDeviceSynchronize");
        let function: Symbol<'_, unsafe extern "C" fn() -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("synchronize device", unsafe { function() })
    }

    fn ipc_get(&self, handle: &mut RawIpcMemHandle, data: *mut c_void) -> Result<(), MemError> {
        let symbol = self.symbol("hipIpcGetMemHandle", "cudaIpcGetMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut RawIpcMemHandle, *mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("export IPC memory handle", unsafe {
            function(handle, data)
        })
    }

    fn ipc_open(&self, data: &mut *mut c_void, handle: RawIpcMemHandle) -> Result<(), MemError> {
        let symbol = self.symbol("hipIpcOpenMemHandle", "cudaIpcOpenMemHandle");
        let function: Symbol<
            '_,
            unsafe extern "C" fn(*mut *mut c_void, RawIpcMemHandle, c_uint) -> c_int,
        > = unsafe { self.load_symbol(symbol)? };
        self.check("open IPC memory handle", unsafe {
            function(data, handle, IPC_LAZY_ENABLE_PEER_ACCESS)
        })
    }

    fn ipc_close(&self, data: *mut c_void) -> Result<(), MemError> {
        let symbol = self.symbol("hipIpcCloseMemHandle", "cudaIpcCloseMemHandle");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("close IPC memory handle", unsafe { function(data) })
    }

    fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        let symbol = self.symbol("hipFree", "cudaFree");
        let function: Symbol<'_, unsafe extern "C" fn(*mut c_void) -> c_int> =
            unsafe { self.load_symbol(symbol)? };
        self.check("free device memory", unsafe { function(data) })
    }

    fn release(&self, data: *mut c_void, release: Release) -> Result<(), MemError> {
        match release {
            Release::Free => self.free(data),
            Release::IpcClose => self.ipc_close(data),
        }
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

    #[test]
    fn ipc_handle_bytes_round_trip() {
        let bytes = std::array::from_fn(|index| index as u8);
        let handle = IpcMemoryHandle::from_bytes(GpuDialect::Hip, 4096, bytes);

        assert_eq!(handle.dialect(), GpuDialect::Hip);
        assert_eq!(handle.byte_len(), 4096);
        assert_eq!(handle.as_bytes(), &bytes);
        assert_eq!(handle.into_bytes(), bytes);
    }
}
