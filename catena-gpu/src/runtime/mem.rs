use std::{ffi::c_int, path::PathBuf, sync::Arc};

use thiserror::Error;

use super::executor::CatenaMem;
use crate::{codegen::GpuDialect, gpu::GpuApi};

#[derive(Debug, Error)]
pub enum MemError {
    #[error("failed to load {dialect:?} GPU library from {paths:?}: {source}")]
    LoadLibrary {
        dialect: GpuDialect,
        paths: Vec<PathBuf>,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to load {dialect:?} GPU symbol `{symbol}`: {source}")]
    LoadSymbol {
        dialect: GpuDialect,
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("{dialect:?} failed to {operation} with status {status}")]
    GpuOperation {
        dialect: GpuDialect,
        operation: &'static str,
        status: c_int,
    },
    #[error("device memory length {byte_len} is not divisible by eight")]
    InvalidU64Length { byte_len: u64 },
    #[error("device memory length {byte_len} cannot be represented on this platform")]
    LengthTooLarge { byte_len: u64 },
}

#[derive(Debug)]
pub struct MemOwn {
    pub(super) abi: CatenaMem,
    gpu: Arc<GpuApi>,
}

impl MemOwn {
    pub(crate) fn from_u64_slice(values: &[u64], gpu: Arc<GpuApi>) -> Result<Self, MemError> {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        let data = gpu.allocate(bytes.len())?;
        if let Err(error) = gpu.copy_to_device(data, bytes) {
            let _ = gpu.free(data);
            return Err(error);
        }
        Ok(Self {
            abi: CatenaMem {
                data,
                len: bytes.len() as u64,
            },
            gpu,
        })
    }

    pub fn to_u64_vec(&self) -> Result<Vec<u64>, MemError> {
        if !self.abi.len.is_multiple_of(8) {
            return Err(MemError::InvalidU64Length {
                byte_len: self.abi.len,
            });
        }
        let byte_len = usize::try_from(self.abi.len).map_err(|_| MemError::LengthTooLarge {
            byte_len: self.abi.len,
        })?;
        let mut values = vec![0_u64; byte_len / 8];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_len) };
        self.gpu.copy_to_host(self.abi.data.cast_const(), bytes)?;
        Ok(values)
    }

    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect()
    }

    pub(super) fn into_abi(mut self) -> CatenaMem {
        CatenaMem {
            data: std::mem::replace(&mut self.abi.data, std::ptr::null_mut()),
            len: self.abi.len,
        }
    }

    pub(super) unsafe fn from_abi(abi: CatenaMem, gpu: Arc<GpuApi>) -> Self {
        Self { abi, gpu }
    }
}

impl Drop for MemOwn {
    fn drop(&mut self) {
        let data = std::mem::replace(&mut self.abi.data, std::ptr::null_mut());
        let _ = self.gpu.free(data);
    }
}
