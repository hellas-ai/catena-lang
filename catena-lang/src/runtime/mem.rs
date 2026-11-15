use std::{
    ffi::{c_int, c_void},
    fs::File,
    io::{Read, Seek},
    marker::PhantomData,
    path::PathBuf,
    sync::Arc,
};

use memmap2::Mmap;
use thiserror::Error;

use super::{GpuDialect, executor::CatenaMem};
use crate::gpu::GpuApi;

#[derive(Debug, Error)]
pub enum MemError {
    #[error("file memory I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("registered file has a null device pointer")]
    InvalidFilePointer,
    #[error("byte range {offset} + {byte_len} exceeds memory length {allocation_byte_len}")]
    InvalidRange {
        offset: u64,
        byte_len: u64,
        allocation_byte_len: u64,
    },
    #[error("registered file memory must be borrowed, not transferred")]
    FileRequiresBorrow,
    #[error(
        "failed to load {dialect:?} GPU library (tried: {tried}): {source}",
        tried = display_paths(tried)
    )]
    LoadLibrary {
        dialect: GpuDialect,
        tried: Vec<PathBuf>,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to resolve {dialect:?} GPU symbol `{symbol}`: {source}")]
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
    #[error("memory length {byte_len} is not a whole number of {element_size}-byte elements")]
    InvalidElementLength { byte_len: u64, element_size: usize },
}

/// Owned device memory. Registered files must be passed to programs by borrowing.
#[derive(Debug)]
pub struct MemOwn {
    pub(super) abi: CatenaMem,
    gpu: Arc<GpuApi>,
    file: Option<RegisteredFile>,
}

#[derive(Debug)]
pub(crate) struct RegisteredFile {
    pub(crate) file: File,
    mapping: Option<Mmap>,
    gpu: Arc<GpuApi>,
}

impl RegisteredFile {
    pub(crate) fn byte_len(&self) -> u64 {
        self.mapping.as_ref().expect("live file mapping").len() as u64
    }
}

impl Drop for RegisteredFile {
    fn drop(&mut self) {
        if let Some(mapping) = self.mapping.take()
            && self
                .gpu
                .unregister_host(mapping.as_ptr().cast_mut().cast())
                .is_err()
        {
            // A failed unregister may leave the driver using these pages.
            std::mem::forget(mapping);
        }
    }
}

/// A borrowed device allocation which can be passed into a Catena program without
/// transferring ownership.
#[derive(Debug, Clone, Copy)]
pub struct MemRef<'a> {
    pub(super) abi: CatenaMem,
    dialect: GpuDialect,
    pub(crate) file: Option<(&'a RegisteredFile, u64)>,
    _lifetime: PhantomData<&'a ()>,
}

impl MemOwn {
    /// Make the entire file available on the current device, independent of its cursor.
    /// Integrated devices register a read-only mapping; discrete devices allocate
    /// and copy through a bounded read buffer. Empty files need no allocation.
    ///
    /// Registered file memory supports [`Self::as_ref`] and [`Self::view`], but
    /// cannot be transferred as an owned program argument. SafeRuntime supports
    /// these borrows on Linux through the live file descriptor in procfs.
    ///
    /// # Safety
    ///
    /// The file must remain immutable (including its length) for this owner's
    /// lifetime, even through other descriptors or processes. GPU programs must
    /// only read the memory, and all GPU use must finish before this owner drops.
    pub unsafe fn from_file(mut file: File, dialect: GpuDialect) -> Result<Self, MemError> {
        let byte_len = file.metadata()?.len();
        let len = usize::try_from(byte_len)
            .ok()
            .filter(|&len| len <= isize::MAX as usize)
            .ok_or(MemError::LengthTooLarge { byte_len })?;
        let gpu = GpuApi::load(dialect)?;
        if len != 0 && gpu.is_integrated()? {
            // SAFETY: the caller guarantees that the file remains immutable.
            let mapping = unsafe { Mmap::map(&file)? };
            let host = mapping.as_ptr().cast_mut().cast();
            gpu.register_host(host, len)?;
            let backing = RegisteredFile {
                file,
                mapping: Some(mapping),
                gpu: gpu.clone(),
            };
            let data = gpu.host_device_pointer(host)?;
            Ok(Self {
                abi: CatenaMem {
                    data,
                    len: byte_len,
                },
                gpu,
                file: Some(backing),
            })
        } else {
            file.rewind()?;
            let data = gpu.allocate(len)?;
            // SAFETY: allocated by this GPU API, with ownership held here.
            let memory = unsafe { Self::from_raw_parts_with_gpu(data, byte_len, gpu) };
            let mut buffer = vec![0_u8; len.min(1024 * 1024)];
            let mut offset = 0;
            while offset < len {
                let count = buffer.len().min(len - offset);
                file.read_exact(&mut buffer[..count])?;
                memory
                    .gpu
                    .copy_host_to_device(data.wrapping_byte_add(offset), &buffer[..count])?;
                offset += count;
            }
            if file.read(&mut [0])? != 0 {
                return Err(std::io::Error::other("file length changed while loading").into());
            }
            Ok(memory)
        }
    }

    /// Copy a slice into a new device allocation.
    pub fn from_u64_slice(values: &[u64], dialect: GpuDialect) -> Result<Self, MemError> {
        Self::from_bytes(slice_as_bytes(values), dialect)
    }

    /// Copy a slice into a new device allocation.
    pub fn from_u16_slice(values: &[u16], dialect: GpuDialect) -> Result<Self, MemError> {
        Self::from_bytes(slice_as_bytes(values), dialect)
    }

    /// Copy a slice into a new device allocation.
    pub fn from_f32_slice(values: &[f32], dialect: GpuDialect) -> Result<Self, MemError> {
        Self::from_bytes(slice_as_bytes(values), dialect)
    }

    fn from_bytes(bytes: &[u8], dialect: GpuDialect) -> Result<Self, MemError> {
        let gpu = GpuApi::load(dialect)?;
        let data = gpu.allocate(bytes.len())?;
        // SAFETY: `data` was just allocated by this same GPU API.
        let mut memory = unsafe { Self::from_raw_parts_with_gpu(data, bytes.len() as u64, gpu) };
        memory.write_from_host(bytes)?;
        Ok(memory)
    }

    /// Take ownership of an existing device allocation.
    ///
    /// This is the explicit boundary used by external memory managers. Runtime
    /// does not inspect the pointer or try to determine whether it aliases any
    /// other memory.
    ///
    /// # Safety
    ///
    /// On success, `data` must be the uniquely owned base pointer of an
    /// allocation created by the allocator for `dialect` (or null when
    /// `byte_len` is zero). It must be valid to transfer to generated Catena
    /// code and to release with `hipFree` or `cudaFree` for that dialect.
    /// Imported IPC mappings therefore must not be wrapped as `MemOwn`.
    /// Ownership remains with the caller if loading the GPU runtime fails.
    pub unsafe fn from_raw_parts(
        data: *mut c_void,
        byte_len: u64,
        dialect: GpuDialect,
    ) -> Result<Self, MemError> {
        let gpu = GpuApi::load(dialect)?;
        // SAFETY: upheld by this function's caller.
        Ok(unsafe { Self::from_raw_parts_with_gpu(data, byte_len, gpu) })
    }

    pub fn to_u16_vec(&self) -> Vec<u16> {
        self.try_to_u16_vec()
            .expect("failed to read memory as u16 values")
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.try_to_f32_vec()
            .expect("failed to read memory as f32 values")
    }

    pub fn to_u64_vec(&self) -> Vec<u64> {
        self.try_to_u64_vec()
            .expect("failed to read memory as u64 values")
    }

    /// Read this buffer into host memory, reporting copy and element-size errors.
    pub fn try_to_f32_vec(&self) -> Result<Vec<f32>, MemError> {
        self.try_to_vec()
    }

    /// Read this buffer into host memory, reporting copy and element-size errors.
    pub fn try_to_u16_vec(&self) -> Result<Vec<u16>, MemError> {
        self.try_to_vec()
    }

    /// Read this buffer into host memory, reporting copy and element-size errors.
    pub fn try_to_u64_vec(&self) -> Result<Vec<u64>, MemError> {
        self.try_to_vec()
    }

    fn try_to_vec<T: Copy + Default>(&self) -> Result<Vec<T>, MemError> {
        let element_size = std::mem::size_of::<T>();
        if !self.abi.len.is_multiple_of(element_size as u64) {
            return Err(MemError::InvalidElementLength {
                byte_len: self.abi.len,
                element_size,
            });
        }
        let byte_len = usize::try_from(self.abi.len).map_err(|_| MemError::LengthTooLarge {
            byte_len: self.abi.len,
        })?;
        if byte_len == 0 {
            return Ok(Vec::new());
        }
        let len = byte_len / element_size;
        let mut values = vec![T::default(); len];
        let output =
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_len) };
        self.gpu
            .copy_device_to_host(self.abi.data.cast_const(), output)?;
        Ok(values)
    }

    /// Return the raw device pointer without transferring ownership.
    pub fn as_ptr(&self) -> *mut c_void {
        self.abi.data
    }

    pub fn byte_len(&self) -> u64 {
        self.abi.len
    }

    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect()
    }

    pub fn as_ref(&self) -> MemRef<'_> {
        MemRef {
            abi: self.abi,
            dialect: self.dialect(),
            file: self.file.as_ref().map(|file| (file, 0)),
            _lifetime: PhantomData,
        }
    }

    /// Borrow a checked byte range, retaining its allocation and file provenance.
    pub fn view(&self, offset: u64, byte_len: u64) -> Result<MemRef<'_>, MemError> {
        self.as_ref().view(offset, byte_len)
    }

    pub(crate) fn is_registered_file(&self) -> bool {
        self.file.is_some()
    }

    fn write_from_host(&mut self, bytes: &[u8]) -> Result<(), MemError> {
        debug_assert_eq!(self.abi.len, bytes.len() as u64);
        self.gpu.copy_host_to_device(self.abi.data, bytes)
    }

    /// Wrap a pointer using an already loaded GPU API.
    ///
    /// # Safety
    ///
    /// The same ownership requirements as [`MemOwn::from_raw_parts`] apply,
    /// and `data` must belong to `gpu`'s dialect.
    pub(crate) unsafe fn from_raw_parts_with_gpu(
        data: *mut c_void,
        byte_len: u64,
        gpu: Arc<GpuApi>,
    ) -> Self {
        Self {
            abi: CatenaMem {
                data,
                len: byte_len,
            },
            gpu,
            file: None,
        }
    }

    pub(super) fn into_abi(mut self) -> CatenaMem {
        assert!(
            !self.is_registered_file(),
            "registered files cannot cross the owned ABI"
        );
        CatenaMem {
            data: std::mem::replace(&mut self.abi.data, std::ptr::null_mut()),
            len: self.abi.len,
        }
    }
}

impl Drop for MemOwn {
    fn drop(&mut self) {
        if self.file.take().is_some() {
            return;
        }
        let data = std::mem::replace(&mut self.abi.data, std::ptr::null_mut());
        let _ = self.gpu.free(data);
    }
}

impl<'a> MemRef<'a> {
    /// Borrow an existing device-memory region for the lifetime of `lease`.
    ///
    /// This constructor confers no ownership and performs no alias analysis.
    ///
    /// # Safety
    ///
    /// `data..data + byte_len` must remain a valid device-memory region for
    /// `dialect` until `lease` is no longer borrowed. `lease` must guard the
    /// actual lifetime of that region.
    pub unsafe fn from_raw_parts<L: ?Sized>(
        data: *mut c_void,
        byte_len: u64,
        dialect: GpuDialect,
        _lease: &'a L,
    ) -> Self {
        Self {
            abi: CatenaMem {
                data,
                len: byte_len,
            },
            dialect,
            file: None,
            _lifetime: PhantomData,
        }
    }

    /// Return the borrowed raw device pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.abi.data
    }

    pub fn byte_len(&self) -> u64 {
        self.abi.len
    }

    pub fn dialect(&self) -> GpuDialect {
        self.dialect
    }

    /// Borrow a checked subrange of this view.
    pub fn view(&self, offset: u64, byte_len: u64) -> Result<Self, MemError> {
        if offset
            .checked_add(byte_len)
            .is_none_or(|end| end > self.abi.len)
        {
            return Err(MemError::InvalidRange {
                offset,
                byte_len,
                allocation_byte_len: self.abi.len,
            });
        }
        let offset_usize =
            usize::try_from(offset).map_err(|_| MemError::LengthTooLarge { byte_len: offset })?;
        Ok(Self {
            abi: CatenaMem {
                data: self.abi.data.wrapping_byte_add(offset_usize),
                len: byte_len,
            },
            file: self.file.map(|(file, base)| (file, base + offset)),
            ..*self
        })
    }
}

fn slice_as_bytes<T>(values: &[T]) -> &[u8] {
    let byte_len = std::mem::size_of_val(values);
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
