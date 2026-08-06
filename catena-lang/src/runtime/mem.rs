use std::{ffi::c_int, path::PathBuf};

use thiserror::Error;

use super::device_mem::{DeviceAllocator, DeviceBuffer};
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
    #[error("memory length {byte_len} is not a whole number of {element_size}-byte elements")]
    InvalidElementLength { byte_len: u64, element_size: usize },
}

/// Mem values represent a device pointer and byte length which can be passed into a Catena program.
#[derive(Debug)]
pub struct Mem {
    pub(crate) abi: CatenaMem,
    owner: MemOwner,
}

#[derive(Debug)]
enum MemOwner {
    Generated(DeviceAllocator),
    Device(DeviceBuffer),
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
        match &self.owner {
            MemOwner::Generated(allocator) => {
                allocator.copy_device_to_host_raw(
                    output.as_mut_ptr().cast(),
                    self.abi.data,
                    byte_len,
                )?;
            }
            MemOwner::Device(device) => device.read_all(output)?,
        }
        Ok(values)
    }

    pub(crate) fn from_device_buffer(device: DeviceBuffer) -> Self {
        let abi = CatenaMem {
            data: device.data(),
            len: device.byte_len() as u64,
        };
        Self {
            abi,
            owner: MemOwner::Device(device),
        }
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        match &self.owner {
            MemOwner::Generated(allocator) => allocator.dialect(),
            MemOwner::Device(device) => device.dialect(),
        }
    }

    pub(crate) fn null(allocator: DeviceAllocator) -> Self {
        Self {
            abi: CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            },
            owner: MemOwner::Generated(allocator),
        }
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        if self.abi.data.is_null() {
            return;
        }
        if let MemOwner::Generated(allocator) = &self.owner {
            let _ = allocator.free_raw(self.abi.data);
        }
    }
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
