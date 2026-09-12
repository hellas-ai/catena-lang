use std::{collections::HashMap, fs::File, os::fd::AsRawFd};

use anyhow::{Context, Result, ensure};
use memmap2::MmapOptions;

use crate::{
    codegen::GpuDialect,
    gpu::{GpuApi, vmm::SharedAllocation},
    runtime::MemRef,
};

pub(super) const MAX_ASSET_BYTES: u64 = 1 << 40;
pub(crate) const MAX_RESIDENT_ASSET_BYTES: u64 = 4 << 40;
pub(crate) const MAX_RESIDENT_ASSETS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResidentAsset {
    pub(super) id: u64,
    pub(super) byte_len: u64,
}

// Owners deduplicate verified content keys. Execution workers retain imported
// mappings under the owner's IDs; both use the same range and capacity checks.
pub(super) struct AssetStore {
    dialect: GpuDialect,
    keys: HashMap<[u8; 32], u64>,
    assets: HashMap<u64, SharedAllocation>,
    resident_bytes: u64,
}

impl AssetStore {
    pub(super) fn new(dialect: GpuDialect) -> Self {
        Self {
            dialect,
            keys: HashMap::new(),
            assets: HashMap::new(),
            resident_bytes: 0,
        }
    }

    pub(super) fn attach(
        &mut self,
        key: [u8; 32],
        byte_len: u64,
        file: File,
    ) -> Result<ResidentAsset> {
        let len = validated_len(&file, byte_len)?;
        if let Some(&id) = self.keys.get(&key) {
            ensure!(
                self.asset(id)?.byte_len() == len,
                "content key is already attached with a different length"
            );
            return Ok(ResidentAsset { id, byte_len });
        }
        let total = self.admit(byte_len)?;
        let bytes =
            unsafe { MmapOptions::new().len(len).map(&file) }.context("failed to mmap asset")?;
        let allocation = GpuApi::load(self.dialect)?.own_shared(&bytes)?;
        let id = self.assets.len() as u64 + 1;
        self.assets.insert(id, allocation);
        self.keys.insert(key, id);
        self.resident_bytes = total;
        Ok(ResidentAsset { id, byte_len })
    }

    pub(super) fn import(
        &mut self,
        id: u64,
        byte_len: u64,
        allocation_len: u64,
        file: File,
    ) -> Result<ResidentAsset> {
        ensure!(id != 0, "invalid resident asset ID");
        if let Some(asset) = self.assets.get(&id) {
            ensure!(
                asset.byte_len() as u64 == byte_len
                    && asset.allocation_len() as u64 == allocation_len,
                "resident asset ID is already imported with a different length"
            );
            return Ok(ResidentAsset { id, byte_len });
        }
        let total = self.admit(byte_len)?;
        let asset = GpuApi::load(self.dialect)?.import_shared(
            file.into(),
            usize::try_from(byte_len)?,
            usize::try_from(allocation_len)?,
        )?;
        self.assets.insert(id, asset);
        self.resident_bytes = total;
        Ok(ResidentAsset { id, byte_len })
    }

    pub(super) fn export(&self, id: u64) -> Result<(File, u64, u64)> {
        let asset = self.asset(id)?;
        Ok((
            asset.export_fd()?.into(),
            asset.byte_len() as u64,
            asset.allocation_len() as u64,
        ))
    }

    pub(super) fn usage(&self) -> (u64, u64, u64) {
        (
            self.assets.len() as u64,
            self.resident_bytes,
            self.assets
                .values()
                .map(|asset| asset.allocation_len() as u64)
                .sum(),
        )
    }

    fn admit(&self, byte_len: u64) -> Result<u64> {
        validate_byte_len(byte_len)?;
        ensure!(
            self.assets.len() < MAX_RESIDENT_ASSETS,
            "asset store already has the maximum of {MAX_RESIDENT_ASSETS} resident assets"
        );
        self.resident_bytes
            .checked_add(byte_len)
            .filter(|total| *total <= MAX_RESIDENT_ASSET_BYTES)
            .context("resident assets exceed the byte limit")
    }

    fn asset(&self, id: u64) -> Result<&SharedAllocation> {
        self.assets
            .get(&id)
            .with_context(|| format!("unknown resident asset {id}"))
    }

    pub(super) fn mem_ref(&self, id: u64, offset: u64, byte_len: u64) -> Result<MemRef<'_>> {
        let asset = self.asset(id)?;
        let end = offset
            .checked_add(byte_len)
            .context("asset slice range overflowed")?;
        ensure!(
            end <= asset.byte_len() as u64,
            "mapped buffer exceeds its asset"
        );
        let data = unsafe {
            asset
                .ptr()
                .cast::<u8>()
                .add(usize::try_from(offset)?)
                .cast()
        };
        Ok(unsafe { MemRef::from_raw_parts(data, byte_len, self.dialect, asset) })
    }
}

fn validated_len(file: &File, expected: u64) -> Result<usize> {
    validate_byte_len(expected)?;
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    if flags == -1 {
        return Err(std::io::Error::last_os_error()).context("failed to inspect asset descriptor");
    }
    ensure!(
        flags & libc::O_ACCMODE == libc::O_RDONLY,
        "asset descriptor must be opened read-only"
    );
    let metadata = file
        .metadata()
        .context("failed to inspect asset descriptor")?;
    ensure!(
        metadata.is_file(),
        "asset descriptor must refer to a regular file"
    );
    ensure!(
        metadata.len() == expected,
        "asset descriptor has {} bytes, expected {expected}",
        metadata.len()
    );
    usize::try_from(expected).context("asset length exceeds usize")
}

pub(super) fn validate_byte_len(byte_len: u64) -> Result<()> {
    ensure!(byte_len != 0, "cannot attach an empty asset");
    ensure!(
        byte_len <= MAX_ASSET_BYTES,
        "asset has {byte_len} bytes, exceeding the {MAX_ASSET_BYTES}-byte limit"
    );
    Ok(())
}
