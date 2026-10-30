//! Compile generated GPU C++ code to a shared object file.

use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::GpuDialect;

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("failed to create temporary build directory or copy generated source: {0}")]
    TempDir(#[from] std::io::Error),
    #[error("GPU compiler `{compiler}` is unavailable: {source}")]
    CompilerUnavailable {
        compiler: String,
        #[source]
        source: std::io::Error,
    },
    #[error("GPU compilation with `{compiler}` failed with status {status}: {stderr}")]
    CompilerFailed {
        compiler: String,
        status: ExitStatus,
        stderr: String,
    },
}

/// A shared object file created by compiling generated Catena GPU C++.
#[derive(Debug)]
pub(super) struct SharedObject {
    _build_dir: tempfile::TempDir,
    path: PathBuf,
}

impl SharedObject {
    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

pub(super) fn compile(cpp_path: &Path, dialect: GpuDialect) -> Result<SharedObject, ArtifactError> {
    compile_with(
        cpp_path,
        dialect,
        Command::new(gpu_compiler(dialect)),
        native_cache_root(),
    )
}

fn compile_with(
    cpp_path: &Path,
    dialect: GpuDialect,
    command: Command,
    cache_root: Option<PathBuf>,
) -> Result<SharedObject, ArtifactError> {
    let build_dir = tempfile::Builder::new()
        .prefix("catena-module-")
        .tempdir()?;
    let module_filename = match dialect {
        GpuDialect::Hip => "module.cpp",
        GpuDialect::Cuda => "module.cu",
    };
    let module_path = build_dir.path().join(module_filename);
    let so_path = build_dir.path().join("module.so");
    std::fs::copy(cpp_path, &module_path)?;

    let build = || compile_uncached(&module_path, &so_path, dialect, command);
    if let Some(root) = cache_root {
        // Hash the private source copy that will actually be compiled.
        let source = fs::read(&module_path)?;
        let key = native_key(&source, dialect, NATIVE_CACHE_VERSION);
        compile_cached(&root, &key, &so_path, build)?;
    } else {
        build()?;
    }

    Ok(SharedObject {
        _build_dir: build_dir,
        path: so_path,
    })
}

fn compile_uncached(
    module_path: &Path,
    so_path: &Path,
    dialect: GpuDialect,
    mut command: Command,
) -> Result<(), ArtifactError> {
    let compiler_display = command.get_program().to_string_lossy().into_owned();
    command.arg("-shared").arg("-O2");
    match dialect {
        GpuDialect::Hip => {
            command
                .arg("-fPIC")
                .arg("--std=c++17")
                // Keep multiply/add as separately rounded operations. This prevents a
                // future generated `a * b + c` expression from being contracted to FMA.
                .arg("-ffp-contract=off")
                // This is the default, but keep it explicit because reproducibility
                // depends on avoiding reassociation and other fast-math transforms.
                .arg("-fno-fast-math");
        }
        GpuDialect::Cuda => {
            command
                .arg("-Xcompiler")
                .arg("-fPIC")
                .arg("--std=c++17")
                // Default to SM_80 (Ampere and later)
                .arg("-arch=sm_80")
                // Match the no-FMA intent for generated arithmetic.
                .arg("--fmad=false");
        }
    }
    let output = command
        .arg(module_path)
        .arg("-o")
        .arg(so_path)
        .output()
        .map_err(|source| ArtifactError::CompilerUnavailable {
            compiler: compiler_display.clone(),
            source,
        })?;

    if !output.status.success() {
        return Err(ArtifactError::CompilerFailed {
            compiler: compiler_display,
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    Ok(())
}

// Bump when compiler flags or native-build behavior change. Toolchain and GPU
// changes require clearing this development cache manually.
const NATIVE_CACHE_VERSION: &[u8] = b"catena-native-v1";

fn native_key(source: &[u8], dialect: GpuDialect, version: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update((version.len() as u64).to_le_bytes());
    hash.update(version);
    hash.update([match dialect {
        GpuDialect::Hip => 0,
        GpuDialect::Cuda => 1,
    }]);
    hash.update(source);
    format!("{:x}", hash.finalize())
}

fn native_cache_root() -> Option<PathBuf> {
    if std::env::var_os("CATENA_GPU_CACHE").is_some_and(|v| v == "0")
        || std::env::var_os("CATENA_NATIVE_CACHE").is_some_and(|v| v == "0")
    {
        return None;
    }
    super::module_cache::cache_root().map(|root| root.join("native"))
}

fn diagnostic(key: &str, state: &str) {
    if std::env::var_os("CATENA_COMPILE_TIMINGS").is_some() {
        eprintln!("[compile] Native cache {state}: {key}");
    }
}

fn compile_cached(
    root: &Path,
    key: &str,
    so_path: &Path,
    build: impl FnOnce() -> Result<(), ArtifactError>,
) -> Result<(), ArtifactError> {
    let guard = (|| -> io::Result<fs::File> {
        fs::create_dir_all(root)?;
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(format!("{key}.lock")))?;
        file.lock()?;
        Ok(file)
    })();
    let _guard = match guard {
        Ok(guard) => guard,
        Err(error) => {
            diagnostic(key, &format!("unavailable: {error}"));
            return build();
        }
    };
    let cached_path = root.join(format!("{key}.so"));
    match fs::copy(&cached_path, so_path) {
        Ok(_) => {
            diagnostic(key, "hit");
            return Ok(());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => diagnostic(key, "miss"),
        Err(error) => diagnostic(key, &format!("could not read, compiling: {error}")),
    }
    build()?;
    let stored = (|| -> io::Result<()> {
        let tmp = tempfile::NamedTempFile::new_in(root)?;
        fs::copy(so_path, tmp.path())?;
        tmp.persist(cached_path).map_err(io::Error::other)?;
        Ok(())
    })();
    if let Err(error) = stored {
        diagnostic(key, &format!("could not store: {error}"));
    }
    Ok(())
}

fn gpu_compiler(dialect: GpuDialect) -> OsString {
    match dialect {
        GpuDialect::Hip => OsString::from("hipcc"),
        GpuDialect::Cuda => OsString::from("nvcc"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        dir: tempfile::TempDir,
        compiler: PathBuf,
        source: PathBuf,
        cache: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let compiler = dir.path().join("compiler");
            fs::write(
                &compiler,
                r#"#!/bin/sh
echo compile >> "$0.calls"
if [ -f "$0.fail" ]; then
    echo 'deliberate failure' >&2
    exit 1
fi
while [ "$#" -gt 0 ]; do
    case "$1" in
        *.cpp|*.cu) source="$1" ;;
        -o) shift; output="$1" ;;
    esac
    shift
done
cat "$source" > "$output"
"#,
            )
            .unwrap();
            let source = dir.path().join("input.cpp");
            fs::write(&source, b"generated source").unwrap();
            let cache = dir.path().join("native");
            Self {
                dir,
                compiler,
                source,
                cache,
            }
        }

        fn command(&self) -> Command {
            let mut command = Command::new("/bin/sh");
            command.arg(&self.compiler);
            command
        }

        fn compile(&self) -> Result<SharedObject, ArtifactError> {
            compile_with(
                &self.source,
                GpuDialect::Hip,
                self.command(),
                Some(self.cache.clone()),
            )
        }

        fn calls(&self) -> usize {
            fs::read_to_string(self.compiler.with_extension("calls"))
                .unwrap()
                .lines()
                .count()
        }
    }

    #[test]
    fn warm_load_skips_compiler_and_uses_private_paths() {
        let fixture = Fixture::new();
        let first = fixture.compile().unwrap();
        let second = fixture.compile().unwrap();
        assert_eq!(fixture.calls(), 1);
        assert_ne!(first.path(), second.path());
        assert_eq!(
            fs::read(first.path()).unwrap(),
            fs::read(second.path()).unwrap()
        );
        drop(first);
        assert_eq!(fs::read(second.path()).unwrap(), b"generated source");

        fs::write(&fixture.source, b"changed source").unwrap();
        let changed = fixture.compile().unwrap();
        assert_eq!(fixture.calls(), 2);
        assert_eq!(fs::read(changed.path()).unwrap(), b"changed source");
        compile_with(
            &fixture.source,
            GpuDialect::Cuda,
            fixture.command(),
            Some(fixture.cache.clone()),
        )
        .unwrap();
        assert_eq!(fixture.calls(), 3);
        assert_ne!(
            native_key(b"source", GpuDialect::Hip, NATIVE_CACHE_VERSION),
            native_key(b"source", GpuDialect::Hip, b"next-version")
        );
    }

    #[test]
    fn competing_requests_compile_once() {
        let fixture = Fixture::new();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let fixture = &fixture;
                scope.spawn(move || {
                    fixture.compile().unwrap();
                });
            }
        });
        assert_eq!(fixture.calls(), 1);
    }

    #[test]
    fn failures_retry_and_unavailable_storage_or_disabled_cache_still_compile() {
        let mut fixture = Fixture::new();
        let fail = fixture.compiler.with_extension("fail");
        fs::write(&fail, b"").unwrap();
        for _ in 0..2 {
            assert!(matches!(
                fixture.compile(),
                Err(ArtifactError::CompilerFailed { .. })
            ));
        }
        assert_eq!(fixture.calls(), 2);
        assert!(
            !fs::read_dir(&fixture.cache).unwrap().any(|p| p
                .unwrap()
                .path()
                .extension()
                .is_some_and(|ext| ext == "so"))
        );
        fs::remove_file(fail).unwrap();
        fixture.compile().unwrap();
        assert_eq!(fixture.calls(), 3);

        fixture.cache = fixture.dir.path().join("file");
        fs::write(&fixture.cache, b"not a directory").unwrap();
        fixture.compile().unwrap();
        assert_eq!(fixture.calls(), 4);
        for _ in 0..2 {
            compile_with(&fixture.source, GpuDialect::Hip, fixture.command(), None).unwrap();
        }
        assert_eq!(fixture.calls(), 6);
    }

    #[test]
    fn environment_switches() {
        const CHILD: &str = "CATENA_TEST_NATIVE_CACHE_ENABLED";
        if let Some(expected) = std::env::var_os(CHILD) {
            let root = native_cache_root();
            assert_eq!(root.is_some(), expected == "1");
            if let Some(root) = root {
                assert_eq!(
                    root,
                    PathBuf::from(std::env::var_os("CATENA_GPU_CACHE_DIR").unwrap()).join("native")
                );
            }
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        for (global, native, hex, enabled) in [
            ("1", "1", "1", "1"),
            ("0", "1", "1", "0"),
            ("1", "0", "1", "0"),
            ("1", "1", "0", "1"),
        ] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "runtime::artifact::tests::environment_switches"])
                .env("CATENA_GPU_CACHE_DIR", dir.path())
                .env("CATENA_GPU_CACHE", global)
                .env("CATENA_NATIVE_CACHE", native)
                .env("CATENA_HEX_CACHE", hex)
                .env(CHILD, enabled)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
    }
}
