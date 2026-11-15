//! Four isolated cases: HIP/CUDA × integrated/discrete, ranges, and mapping cleanup.
#![cfg(target_os = "linux")]
use catena_lang::runtime::{GpuDialect, MemOwn};
use std::{env, fs, io::Write, process::Command};

#[test]
fn file_memory() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(dialect) = env::var("FILE_DIALECT") else {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("mock.c");
        fs::write(&source, include_str!("file_memory_mock.c"))?;
        let library = dir.path().join("libamdhip64.so");
        assert!(
            Command::new(env::var_os("CC").unwrap_or("cc".into()))
                .args(["-shared", "-fPIC", "-o"])
                .arg(&library)
                .arg(source)
                .status()?
                .success()
        );
        for name in ["libcudart.so", "libcuda.so.1"] {
            fs::copy(&library, dir.path().join(name))?;
        }
        for dialect in ["hip", "cuda"] {
            for integrated in ["0", "1"] {
                assert!(
                    Command::new(env::current_exe()?)
                        .args(["--exact", "file_memory"])
                        .env("LD_LIBRARY_PATH", dir.path())
                        .env("FILE_DIALECT", dialect)
                        .env("FILE_INTEGRATED", integrated)
                        .status()?
                        .success()
                );
            }
        }
        return Ok(());
    };
    let (dialect, library) = match dialect.as_str() {
        "hip" => (GpuDialect::Hip, "libamdhip64.so"),
        _ => (GpuDialect::Cuda, "libcudart.so"),
    };
    let mut file = tempfile::tempfile()?;
    let bytes: Vec<u8> = (0..1024 * 1024 + 1).map(|i| (i % 251) as u8).collect();
    file.write_all(&bytes)?;
    // The file is immutable, and only the mock exposes host-addressable device pointers.
    let memory = unsafe { MemOwn::from_file(file, dialect)? };
    assert_eq!(memory.byte_len(), bytes.len() as u64);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(memory.as_ptr().cast::<u8>(), bytes.len()) },
        bytes
    );
    assert_eq!(
        memory.view(7, 3).unwrap().as_ptr(),
        memory.as_ptr().wrapping_byte_add(7)
    );
    assert!(memory.view(memory.byte_len(), 0).is_some());
    assert!(memory.view(memory.byte_len(), 1).is_none());
    assert!(memory.view(u64::MAX, 1).is_none());
    drop(memory);
    let library = unsafe { libloading::Library::new(library)? };
    let check: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { library.get(b"file_test_check")? };
    unsafe {
        check();
    }
    Ok(())
}
