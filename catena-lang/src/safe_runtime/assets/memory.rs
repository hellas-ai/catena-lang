use std::{ffi::c_void, fs::File, os::fd::AsRawFd, ptr::NonNull, sync::Arc};

use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::{codegen::GpuDialect, gpu::GpuApi, runtime::MemRef};

pub(super) struct LoadedAsset {
    gpu: Arc<GpuApi>,
    device_base: NonNull<c_void>,
    bytes: AssetMapping,
}

impl LoadedAsset {
    pub(super) fn open(file: File, byte_len: usize, dialect: GpuDialect) -> Result<Self> {
        let gpu = GpuApi::load(dialect)?;
        let read_only = gpu.supports_read_only_host()?;
        // CUDA devices without read-only registration need writable pages.
        // MAP_PRIVATE keeps writes isolated from the source file. Pinning may
        // materialize private host pages, charged to the worker's memory limit.
        let bytes = if read_only {
            AssetMapping::ReadOnly(
                unsafe { MmapOptions::new().len(byte_len).map(&file) }
                    .context("failed to mmap asset")?,
            )
        } else {
            AssetMapping::Private(
                unsafe { MmapOptions::new().len(byte_len).map_copy(&file) }
                    .context("failed to privately mmap asset")?,
            )
        };
        // SAFETY: this object retains the mapping and unregisters before drop.
        let device_base =
            unsafe { gpu.register_host(bytes.as_ptr().cast_mut().cast(), bytes.len(), read_only)? };
        Ok(Self {
            gpu,
            device_base,
            bytes,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(super) fn mem_ref(
        &self,
        offset: usize,
        byte_len: u64,
        dialect: GpuDialect,
    ) -> Result<MemRef<'_>> {
        let byte_len_usize = usize::try_from(byte_len).context("buffer length exceeds usize")?;
        let end = offset
            .checked_add(byte_len_usize)
            .context("mapped buffer range overflowed")?;
        ensure!(end <= self.bytes.len(), "mapped buffer exceeds its asset");
        let data = unsafe {
            self.device_base
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<c_void>()
        };
        Ok(unsafe { MemRef::from_raw_parts(data, byte_len, dialect, self) })
    }
}

pub(super) fn validated_len(file: &File, expected: u64) -> Result<usize> {
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error()).context("failed to inspect asset descriptor");
    }
    ensure!(
        flags & libc::O_ACCMODE == libc::O_RDONLY,
        "asset descriptor must be opened read-only"
    );
    let actual = file
        .metadata()
        .context("failed to inspect asset descriptor")?
        .len();
    ensure!(
        actual == expected,
        "asset descriptor has {actual} bytes, expected {expected}"
    );
    usize::try_from(actual).context("asset length exceeds usize")
}

impl Drop for LoadedAsset {
    fn drop(&mut self) {
        // SAFETY: bytes is still mapped and was registered by this GPU API.
        if let Err(error) = unsafe {
            self.gpu
                .unregister_host(self.bytes.as_ptr().cast_mut().cast())
        } {
            eprintln!("fatal GPU asset cleanup failure: {error}");
            std::process::abort();
        }
    }
}

// Both variants expose only immutable CPU access after registration.
enum AssetMapping {
    ReadOnly(Mmap),
    Private(MmapMut),
}

impl std::ops::Deref for AssetMapping {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::ReadOnly(bytes) => bytes,
            Self::Private(bytes) => bytes,
        }
    }
}
