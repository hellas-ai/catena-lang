use std::{ffi::c_void, fs::File, os::fd::AsRawFd, ptr::NonNull};

use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};

use crate::{codegen::GpuDialect, runtime::MemRef};

use super::hip::HipRegistration;

pub(super) struct LoadedAsset {
    registration: DeviceRegistration,
    bytes: Mmap,
}

impl LoadedAsset {
    pub(super) fn open(file: File, byte_len: usize, dialect: GpuDialect) -> Result<Self> {
        let bytes = unsafe { MmapOptions::new().len(byte_len).map(&file) }
            .context("failed to mmap asset")?;
        let registration = DeviceRegistration::new(&bytes, dialect)?;
        Ok(Self {
            registration,
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
            self.registration
                .device_base()
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

enum DeviceRegistration {
    Hip(HipRegistration),
}

impl DeviceRegistration {
    fn new(bytes: &Mmap, dialect: GpuDialect) -> Result<Self> {
        match dialect {
            GpuDialect::Hip => Ok(Self::Hip(HipRegistration::new(bytes)?)),
            GpuDialect::Cuda => bail!("CUDA-visible mmap assets are not yet supported"),
        }
    }

    fn device_base(&self) -> NonNull<c_void> {
        match self {
            Self::Hip(registration) => registration.device_base,
        }
    }
}
