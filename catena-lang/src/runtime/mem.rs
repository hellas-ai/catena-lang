use std::{ffi::c_int, path::PathBuf};

use thiserror::Error;

use super::device_mem::{DeviceBuffer, IpcMemoryHandle};
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
    #[error("{dialect:?} runtime failed to {operation} with status {status}")]
    GpuOperation {
        dialect: GpuDialect,
        operation: &'static str,
        status: c_int,
    },
    #[error("device memory range {offset}..{end} exceeds allocation length {byte_len}")]
    OutOfBounds {
        offset: usize,
        end: usize,
        byte_len: usize,
    },
    #[error("device memory range overflows: offset {offset}, length {length}")]
    RangeOverflow { offset: usize, length: usize },
    #[error("device memory length {byte_len} cannot be represented on this platform")]
    LengthTooLarge { byte_len: u64 },
    #[error("IPC handle uses {handle_dialect:?}, but the allocator uses {allocator_dialect:?}")]
    DialectMismatch {
        allocator_dialect: GpuDialect,
        handle_dialect: GpuDialect,
    },
    #[error("an imported IPC mapping cannot be exported again")]
    CannotExportImported,
    #[error("device allocation is still shared")]
    AllocationShared,
    #[error("memory view does not belong to its backing device allocation")]
    InvalidView,
    #[error("memory length {byte_len} is not a whole number of {element_size}-byte elements")]
    InvalidElementLength { byte_len: u64, element_size: usize },
    #[error("SafeRuntime child returned invalid memory metadata: {0}")]
    InvalidRemoteMemory(String),
    #[error("IPC memory handle has {actual} bytes, expected 64")]
    InvalidIpcHandleLength { actual: usize },
}

/// Mem values represent a device pointer and byte length which can be passed into a Catena program.
#[derive(Debug, Clone)]
pub struct Mem {
    pub(crate) abi: CatenaMem,
    buffer: DeviceBuffer,
}

impl Mem {
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
        let offset = self.buffer.view_offset(self.abi.data, self.abi.len)?;
        self.buffer.read(offset as usize, output)?;
        Ok(values)
    }

    pub(crate) fn from_device_buffer(buffer: DeviceBuffer) -> Self {
        let abi = CatenaMem {
            data: buffer.data(),
            len: buffer.byte_len() as u64,
        };
        Self { abi, buffer }
    }

    pub(crate) fn from_device_buffer_view(
        buffer: DeviceBuffer,
        offset: u64,
        byte_len: u64,
    ) -> Result<Self, MemError> {
        let data = buffer.data_for_view(offset, byte_len)?;
        Ok(Self {
            abi: CatenaMem {
                data,
                len: byte_len,
            },
            buffer,
        })
    }

    pub(crate) fn from_abi(buffer: DeviceBuffer, abi: CatenaMem) -> Result<Self, MemError> {
        buffer.view_offset(abi.data, abi.len)?;
        Ok(Self { abi, buffer })
    }

    pub fn byte_len(&self) -> u64 {
        self.abi.len
    }

    pub(crate) fn device_dialect(&self) -> GpuDialect {
        self.buffer.dialect()
    }

    pub(crate) fn device_buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }

    pub(crate) fn view(&self, offset: u64, byte_len: u64) -> Result<Self, MemError> {
        Self::from_device_buffer_view(self.buffer.clone(), offset, byte_len)
    }

    pub(crate) fn export_ipc_view(&self) -> Result<(IpcMemoryHandle, u64), MemError> {
        let offset = self.buffer.view_offset(self.abi.data, self.abi.len)?;
        Ok((self.buffer.export_ipc()?, offset))
    }
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
