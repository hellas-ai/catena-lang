//! Process-local CUDA/HIP IPC primitives.
//!
//! This module deliberately does not define a transport protocol. Device
//! addresses are used only while deriving a view offset and are never part of
//! the exported representation.

use std::{ffi::c_void, sync::Arc};

use crate::{
    codegen::GpuDialect,
    gpu::{GpuApi, IPC_HANDLE_BYTES, RawIpcMemHandle},
    runtime::{MemError, MemOwn, MemRef, Runtime},
};

/// Opaque identity for one allocation generation.
///
/// CUDA and HIP guarantee that freeing an allocation and reusing its device
/// address produces a different IPC handle. Device addresses are therefore not
/// part of this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IpcMemoryHandle {
    bytes: [u8; IPC_HANDLE_BYTES],
}

impl IpcMemoryHandle {
    pub(super) fn from_bytes(bytes: [u8; IPC_HANDLE_BYTES]) -> Self {
        Self { bytes }
    }

    pub(super) fn as_bytes(&self) -> &[u8; IPC_HANDLE_BYTES] {
        &self.bytes
    }
}

/// IPC capability and bounds for a device allocation view.
///
/// A `MemOwn` is exported through its borrowed view; ownership itself never
/// crosses the process boundary through the handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExportedIpcView {
    handle: Option<IpcMemoryHandle>,
    allocation_byte_len: u64,
    view_offset: u64,
    byte_len: u64,
}

impl ExportedIpcView {
    fn empty() -> Self {
        Self {
            handle: None,
            allocation_byte_len: 0,
            view_offset: 0,
            byte_len: 0,
        }
    }

    pub(super) fn handle(&self) -> Option<&IpcMemoryHandle> {
        self.handle.as_ref()
    }

    pub(super) fn allocation_byte_len(&self) -> u64 {
        self.allocation_byte_len
    }

    pub(super) fn view_offset(&self) -> u64 {
        self.view_offset
    }

    pub(super) fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

#[derive(Debug, Clone)]
pub(super) struct IpcTransport {
    gpu: Arc<GpuApi>,
}

impl IpcTransport {
    pub(super) fn load(dialect: GpuDialect) -> Result<Self, MemError> {
        Ok(Self {
            gpu: GpuApi::load(dialect)?,
        })
    }

    pub(super) fn from_runtime(runtime: &Runtime) -> Self {
        Self {
            gpu: runtime.gpu_api(),
        }
    }

    pub(super) fn dialect(&self) -> GpuDialect {
        self.gpu.dialect()
    }

    pub(super) fn synchronize(&self) -> Result<(), MemError> {
        self.gpu.synchronize()
    }

    /// Exports a view while its owning process keeps the allocation alive.
    pub(super) fn export_view(&self, memory: MemRef<'_>) -> Result<ExportedIpcView, MemError> {
        debug_assert_eq!(memory.dialect(), self.dialect());
        if memory.byte_len() == 0 {
            return Ok(ExportedIpcView::empty());
        }

        let (base, allocation_byte_len) = self.gpu.address_range(memory.as_ptr().cast_const())?;
        let view_offset = memory.as_ptr() as usize - base as usize;
        let handle = self.gpu.ipc_get_mem_handle(base)?;
        Ok(ExportedIpcView {
            handle: Some(IpcMemoryHandle::from_bytes(handle.bytes)),
            allocation_byte_len: allocation_byte_len as u64,
            view_offset: view_offset as u64,
            byte_len: memory.byte_len(),
        })
    }

    /// Opens a sender-owned allocation for borrowing or copying in this process.
    pub(super) fn import_allocation(
        &self,
        handle: Option<IpcMemoryHandle>,
        allocation_byte_len: u64,
    ) -> Result<ImportedIpcAllocation, MemError> {
        let data = match handle {
            Some(handle) => self.gpu.ipc_open_mem_handle(RawIpcMemHandle {
                bytes: handle.bytes,
            })?,
            None => std::ptr::null_mut(),
        };
        Ok(ImportedIpcAllocation {
            data,
            byte_len: allocation_byte_len,
            gpu: self.gpu.clone(),
        })
    }
}

/// A process-local mapping of an allocation still owned by the sender.
///
/// Owned values are copied out before this mapping is closed rather than
/// turning the imported mapping itself into a `MemOwn`.
#[derive(Debug)]
pub(super) struct ImportedIpcAllocation {
    data: *mut c_void,
    byte_len: u64,
    gpu: Arc<GpuApi>,
}

impl ImportedIpcAllocation {
    pub(super) fn as_mem_ref(&self, view_offset: u64, byte_len: u64) -> Option<MemRef<'_>> {
        let view_end = view_offset.checked_add(byte_len)?;
        if view_end > self.byte_len || (byte_len != 0 && self.data.is_null()) {
            return None;
        }
        let view_offset = usize::try_from(view_offset).ok()?;
        let data = if byte_len == 0 {
            std::ptr::null_mut()
        } else {
            unsafe { self.data.cast::<u8>().add(view_offset).cast::<c_void>() }
        };
        // SAFETY: this value borrows the allocation that owns the imported mapping.
        Some(unsafe { MemRef::from_raw_parts(data, byte_len, self.gpu.dialect(), self) })
    }

    /// Copies an imported view into a new allocation owned by this process.
    pub(super) fn copy_view_into_owned(
        &self,
        view_offset: u64,
        byte_len: u64,
    ) -> Result<Option<MemOwn>, MemError> {
        let Some(source) = self.as_mem_ref(view_offset, byte_len) else {
            return Ok(None);
        };
        let byte_len =
            usize::try_from(byte_len).map_err(|_| MemError::LengthTooLarge { byte_len })?;
        let data = self.gpu.allocate(byte_len)?;
        // SAFETY: `data` was just allocated by this same GPU API.
        let memory =
            unsafe { MemOwn::from_raw_parts_with_gpu(data, byte_len as u64, self.gpu.clone()) };
        self.gpu
            .copy_device_to_device(memory.as_ptr(), source.as_ptr().cast_const(), byte_len)?;
        self.gpu.synchronize()?;
        Ok(Some(memory))
    }
}

impl Drop for ImportedIpcAllocation {
    fn drop(&mut self) {
        let data = std::mem::replace(&mut self.data, std::ptr::null_mut());
        let _ = self.gpu.ipc_close_mem_handle(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_handle_preserves_all_bytes() {
        let bytes = std::array::from_fn(|index| index as u8);
        let handle = IpcMemoryHandle::from_bytes(bytes);
        assert_eq!(handle.as_bytes(), &bytes);
    }

    #[test]
    fn empty_view_needs_no_handle() {
        let view = ExportedIpcView::empty();
        assert!(view.handle().is_none());
        assert_eq!(view.allocation_byte_len(), 0);
        assert_eq!(view.view_offset(), 0);
        assert_eq!(view.byte_len(), 0);
    }
}
