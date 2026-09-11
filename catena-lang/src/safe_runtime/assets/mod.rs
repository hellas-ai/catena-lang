use std::{collections::HashMap, fs::File};

use anyhow::{Result, ensure};

use crate::{codegen::GpuDialect, runtime::MemRef};

mod memory;

use memory::{LoadedAsset, validated_len};

pub(super) const MAX_ASSET_BYTES: u64 = 1 << 40;
pub(crate) const MAX_RESIDENT_ASSET_BYTES: u64 = 4 << 40;
pub(crate) const MAX_RESIDENT_ASSETS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResidentAsset {
    pub(super) id: u64,
    pub(super) byte_len: u64,
}

pub(super) struct AssetStore {
    dialect: GpuDialect,
    keys: HashMap<[u8; 32], usize>,
    assets: Vec<LoadedAsset>,
    resident_bytes: u64,
}

impl AssetStore {
    pub(super) fn new(dialect: GpuDialect) -> Self {
        Self {
            dialect,
            keys: HashMap::new(),
            assets: Vec::new(),
            resident_bytes: 0,
        }
    }

    pub(super) fn attach(
        &mut self,
        key: [u8; 32],
        byte_len: u64,
        file: File,
    ) -> Result<ResidentAsset> {
        validate_byte_len(byte_len)?;
        if let Some(&index) = self.keys.get(&key) {
            let loaded = &self.assets[index];
            ensure!(
                loaded.len() as u64 == byte_len,
                "content key is already attached with {} bytes, not {byte_len}",
                loaded.len()
            );
            validated_len(&file, byte_len)?;
            return Ok(ResidentAsset {
                id: resident_id(index),
                byte_len,
            });
        }

        ensure!(
            self.assets.len() < MAX_RESIDENT_ASSETS,
            "session already has the maximum of {MAX_RESIDENT_ASSETS} resident assets"
        );
        let resident_bytes = self
            .resident_bytes
            .checked_add(byte_len)
            .filter(|total| *total <= MAX_RESIDENT_ASSET_BYTES)
            .ok_or_else(|| anyhow::anyhow!("resident assets exceed the session byte limit"))?;
        let byte_len_usize = validated_len(&file, byte_len)?;
        let loaded = LoadedAsset::open(file, byte_len_usize, self.dialect)?;
        // Validate that the registered mapping is representable at Catena's
        // borrowed-memory boundary without exposing that boundary publicly.
        let _ = loaded.mem_ref(0, byte_len, self.dialect)?;

        let index = self.assets.len();
        self.assets.push(loaded);
        self.keys.insert(key, index);
        self.resident_bytes = resident_bytes;
        Ok(ResidentAsset {
            id: resident_id(index),
            byte_len,
        })
    }

    pub(super) fn mem_ref(&self, id: u64, offset: u64, byte_len: u64) -> Result<MemRef<'_>> {
        let index = usize::try_from(id)
            .ok()
            .and_then(|id| id.checked_sub(1))
            .ok_or_else(|| anyhow::anyhow!("unknown resident asset {id}"))?;
        let asset = self
            .assets
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("unknown resident asset {id}"))?;
        let offset = usize::try_from(offset)
            .map_err(|_| anyhow::anyhow!("asset slice offset exceeds usize"))?;
        asset.mem_ref(offset, byte_len, self.dialect)
    }
}

fn resident_id(index: usize) -> u64 {
    u64::try_from(index)
        .expect("resident asset count is bounded below u64")
        .checked_add(1)
        .expect("resident asset ID space exhausted")
}

pub(super) fn validate_byte_len(byte_len: u64) -> Result<()> {
    ensure!(byte_len != 0, "cannot attach an empty asset");
    ensure!(
        byte_len <= MAX_ASSET_BYTES,
        "asset has {byte_len} bytes, exceeding the {MAX_ASSET_BYTES}-byte limit"
    );
    Ok(())
}
