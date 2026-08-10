use std::{
    ffi::{c_int, c_void},
    marker::PhantomData,
    path::PathBuf,
    sync::Arc,
};

use thiserror::Error;

use super::{executor::CatenaMem, gpu_api::GpuApi};
use crate::codegen::GpuDialect;

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

/// A uniquely owned device allocation which can be transferred into a Catena program.
#[derive(Debug)]
pub struct MemOwn {
    pub(super) abi: CatenaMem,
    gpu: Arc<GpuApi>,
}

/// A borrowed device allocation which can be passed into a Catena program without
/// transferring ownership.
#[derive(Debug, Clone, Copy)]
pub struct MemRef<'a> {
    pub(super) abi: CatenaMem,
    dialect: GpuDialect,
    _lifetime: PhantomData<&'a ()>,
}

impl MemOwn {
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
        let gpu = Arc::new(GpuApi::load(dialect)?);
        // SAFETY: upheld by this function's caller.
        Ok(unsafe { Self::from_raw_parts_with_gpu(data, byte_len, gpu) })
    }

    pub fn to_bf16_vec(&self) -> Vec<half::bf16> {
        self.try_to_bf16_vec()
            .expect("failed to read memory as bf16 values")
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

    /// Read this buffer as BF16 values, reporting copy and element-size errors.
    pub fn try_to_bf16_vec(&self) -> Result<Vec<half::bf16>, MemError> {
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
            _lifetime: PhantomData,
        }
    }

    pub(super) fn write_from_host(&mut self, bytes: &[u8]) -> Result<(), MemError> {
        debug_assert_eq!(self.abi.len, bytes.len() as u64);
        self.gpu.copy_host_to_device(self.abi.data, bytes)
    }

    /// Wrap a pointer using an already loaded GPU API.
    ///
    /// # Safety
    ///
    /// The same ownership requirements as [`MemOwn::from_raw_parts`] apply,
    /// and `data` must belong to `gpu`'s dialect.
    pub(super) unsafe fn from_raw_parts_with_gpu(
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
        }
    }

    pub(super) fn into_abi(mut self) -> CatenaMem {
        CatenaMem {
            data: std::mem::replace(&mut self.abi.data, std::ptr::null_mut()),
            len: self.abi.len,
        }
    }
}

impl Drop for MemOwn {
    fn drop(&mut self) {
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
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
