//! Persistent cache for checked, lowered GPU source and its complete public ABI.
use super::{GpuDialect, RuntimeModule};
use crate::codegen::CodegenKind;
use bincode::Options;
use sha2::{Digest, Sha256};
use std::{fs, io, path::PathBuf};

/// Cache the result of Hex compilation, before native GPU compilation.
///
/// Keys include ordered source texts, backend, dialect, and a build-time compiler
/// fingerprint (including dirty compiler sources and embedded standard library).
/// Set `CATENA_HEX_CACHE=0` or `CATENA_GPU_CACHE=0` to disable caching. Entries live
/// under `$CATENA_GPU_CACHE_DIR/hex`, or `$XDG_CACHE_HOME/catena/gpu/hex` (falling
/// back to `$HOME/.cache`). `CATENA_COMPILE_TIMINGS=1` reports hits and misses.
///
/// Use [`Self::with_context`] when the build closure applies additional policy,
/// such as choosing exports or transforming sources before compilation.
pub struct ModuleCache {
    root: PathBuf,
    key: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Entry {
    key: String,
    digest: Vec<u8>,
    module: RuntimeModule,
}

impl ModuleCache {
    pub fn from_sources(sources: &[&str], kind: CodegenKind, dialect: GpuDialect) -> Option<Self> {
        if std::env::var_os("CATENA_GPU_CACHE").is_some_and(|v| v == "0")
            || std::env::var_os("CATENA_HEX_CACHE").is_some_and(|v| v == "0")
        {
            return None;
        }
        let base = cache_root()?;
        Some(Self {
            root: base.join("hex"),
            key: source_key(sources, kind, dialect),
        })
    }

    /// Include caller-owned compilation policy in cache identity.
    pub fn with_context(mut self, context: &str) -> Self {
        let mut hash = Sha256::new();
        hash_part(&mut hash, self.key.as_bytes());
        hash_part(&mut hash, context.as_bytes());
        self.key = format!("{:x}", hash.finalize());
        self
    }

    /// Return a cached module or compile and atomically store it. Compilation
    /// failures are never cached; cache I/O failures fall back to compilation.
    /// A per-key file lock serializes competing threads and processes.
    pub fn get_or_compile<E>(
        &self,
        build: impl FnOnce() -> Result<RuntimeModule, E>,
    ) -> Result<RuntimeModule, E> {
        let guard = (|| -> io::Result<fs::File> {
            fs::create_dir_all(&self.root)?;
            let file = fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(self.root.join(format!("{}.lock", self.key)))?;
            file.lock()?;
            Ok(file)
        })();
        let _guard = match guard {
            Ok(g) => g,
            Err(error) => {
                self.diagnostic(&format!("unavailable: {error}"));
                return build();
            }
        };
        if let Ok(module) = self.read() {
            self.diagnostic("hit");
            return Ok(module);
        }
        self.diagnostic("miss");
        let module = build()?;
        if let Err(error) = self.write(&module) {
            self.diagnostic(&format!("could not store: {error}"));
        }
        Ok(module)
    }

    fn diagnostic(&self, state: &str) {
        if std::env::var_os("CATENA_COMPILE_TIMINGS").is_some() {
            eprintln!("[compile] Hex module cache {state}: {}", self.key);
        }
    }

    fn read(&self) -> io::Result<RuntimeModule> {
        let bytes = fs::read(self.root.join(format!("{}.bin", self.key)))?;
        let entry: Entry = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(bytes.len() as u64)
            .reject_trailing_bytes()
            .deserialize(&bytes)
            .map_err(io::Error::other)?;
        let encoded = bincode::serialize(&entry.module).map_err(io::Error::other)?;
        if entry.key != self.key || entry.digest != Sha256::digest(encoded).to_vec() {
            return Err(io::Error::other("invalid Hex cache entry"));
        }
        Ok(entry.module)
    }

    fn write(&self, module: &RuntimeModule) -> io::Result<()> {
        let encoded = bincode::serialize(module).map_err(io::Error::other)?;
        let entry = Entry {
            key: self.key.clone(),
            digest: Sha256::digest(encoded).to_vec(),
            module: module.clone(),
        };
        let tmp = tempfile::NamedTempFile::new_in(&self.root)?;
        fs::write(
            tmp.path(),
            bincode::serialize(&entry).map_err(io::Error::other)?,
        )?;
        tmp.persist(self.root.join(format!("{}.bin", self.key)))
            .map_err(io::Error::other)?;
        Ok(())
    }
}

pub(super) fn cache_root() -> Option<PathBuf> {
    std::env::var_os("CATENA_GPU_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                .map(|p| p.join("catena/gpu"))
        })
}

fn hash_part(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn source_key(sources: &[&str], kind: CodegenKind, dialect: GpuDialect) -> String {
    let mut hash = Sha256::new();
    hash_part(&mut hash, b"catena-hex-module-v1");
    hash_part(&mut hash, env!("CATENA_COMPILER_FINGERPRINT").as_bytes());
    hash_part(&mut hash, format!("{kind:?}/{dialect:?}").as_bytes());
    hash_part(&mut hash, &(sources.len() as u64).to_le_bytes());
    for source in sources {
        hash_part(&mut hash, source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::super::{GeneratedFunction, ValueKind};
    use super::*;

    fn fixture() -> RuntimeModule {
        RuntimeModule::new(
            GpuDialect::Hip,
            "source",
            [GeneratedFunction {
                source_name: "entry".into(),
                symbol: "entry_symbol".into(),
                inputs: vec![ValueKind::U64],
                outputs: vec![ValueKind::U64],
            }],
        )
    }
    fn cache(root: &std::path::Path) -> ModuleCache {
        ModuleCache {
            root: root.to_owned(),
            key: "fixture".into(),
        }
    }
    #[test]
    fn keys_separate_source_order_boundaries_dialect_and_context() {
        let key = |s: &[&str], d| source_key(s, CodegenKind::Default, d);
        let first = key(&["ab", "c"], GpuDialect::Hip);
        for other in [
            key(&["a", "bc"], GpuDialect::Hip),
            key(&["c", "ab"], GpuDialect::Hip),
            key(&["abc"], GpuDialect::Hip),
            key(&["ab", "c"], GpuDialect::Cuda),
        ] {
            assert_ne!(first, other);
        }
        assert_ne!(
            cache(std::path::Path::new(".")).with_context("a").key,
            cache(std::path::Path::new(".")).with_context("b").key
        );
    }
    #[test]
    fn roundtrip_preserves_abi_skips_build_and_rebuilds_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        let module = fixture();
        assert_eq!(
            cache
                .get_or_compile(|| Ok::<_, ()>(module.clone()))
                .unwrap(),
            module
        );
        assert_eq!(
            cache
                .get_or_compile(|| Err::<RuntimeModule, _>(()))
                .unwrap(),
            module
        );
        let path = dir.path().join("fixture.bin");
        let mut entry: Entry = bincode::deserialize(&fs::read(&path).unwrap()).unwrap();
        entry.module.functions[0].inputs.clear(); // ABI is covered by checksum too.
        fs::write(&path, bincode::serialize(&entry).unwrap()).unwrap();
        assert_eq!(
            cache
                .get_or_compile(|| Ok::<_, ()>(module.clone()))
                .unwrap(),
            module
        );
        fs::write(&path, b"corrupt").unwrap();
        assert_eq!(
            cache
                .get_or_compile(|| Ok::<_, ()>(module.clone()))
                .unwrap(),
            module
        );
    }
    #[test]
    fn errors_are_not_cached_and_unavailable_cache_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        assert!(
            cache
                .get_or_compile(|| Err::<RuntimeModule, _>("failed"))
                .is_err()
        );
        assert!(!dir.path().join("fixture.bin").exists());
        let module = fixture();
        assert_eq!(
            cache
                .get_or_compile(|| Ok::<_, ()>(module.clone()))
                .unwrap(),
            module
        );
        let file = dir.path().join("file");
        fs::write(&file, b"not a directory").unwrap();
        assert_eq!(
            super::tests::cache(&file)
                .get_or_compile(|| Ok::<_, ()>(module.clone()))
                .unwrap(),
            module
        );
    }
    #[test]
    fn concurrent_requests_compile_once() {
        let dir = tempfile::tempdir().unwrap();
        let count = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let path = dir.path();
                let count = &count;
                scope.spawn(move || {
                    cache(path)
                        .get_or_compile(|| {
                            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            Ok::<_, ()>(fixture())
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
