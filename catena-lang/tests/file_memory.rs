//! Hardware-independent strategy, bounds, and failure cleanup checks.
#[cfg(target_os = "linux")]
use catena_lang::runtime::{GpuDialect, MemOwn};
#[cfg(target_os = "linux")]
use std::{
    env, fs,
    io::{Seek, SeekFrom, Write},
    process::Command,
};

#[cfg(not(target_os = "linux"))]
fn main() {}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::args().nth(1).as_deref() == Some("--case") {
        return case();
    }
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
            let failures = if integrated == "1" {
                &["", "attribute", "register", "pointer", "null"][..]
            } else {
                &["", "attribute", "copy", "truncate"][..]
            };
            for failure in failures {
                assert!(
                    Command::new(env::current_exe()?)
                        .arg("--case")
                        .env("LD_LIBRARY_PATH", dir.path())
                        .env("FILE_TEST_DIALECT", dialect)
                        .env("FILE_TEST_INTEGRATED", integrated)
                        .env("FILE_TEST_FAILURE", failure)
                        .status()?
                        .success(),
                    "{dialect}, integrated={integrated}, failure={failure}"
                );
            }
        }
    }
    println!("file memory: both dialects, both strategies, bounds and cleanup passed");
    Ok(())
}

#[cfg(target_os = "linux")]
fn case() -> Result<(), Box<dyn std::error::Error>> {
    let hip = env::var("FILE_TEST_DIALECT")? == "hip";
    let dialect = if hip {
        GpuDialect::Hip
    } else {
        GpuDialect::Cuda
    };
    let integrated = env::var("FILE_TEST_INTEGRATED")? == "1";
    let failure = env::var("FILE_TEST_FAILURE")?;
    let mut file = tempfile::NamedTempFile::new()?;
    let bytes: Vec<u8> = (0..3 * 1024 * 1024 + 1).map(|i| (i % 251) as u8).collect();
    file.write_all(&bytes)?;
    file.seek(SeekFrom::End(0))?;
    // This single-threaded test process sets the mock's fault-injection input.
    unsafe {
        env::set_var("FILE_TEST_PATH", file.path());
    }
    let library = unsafe {
        libloading::Library::new(if hip {
            "libamdhip64.so"
        } else {
            "libcudart.so"
        })?
    };
    let stat: libloading::Symbol<'_, unsafe extern "C" fn(i32) -> usize> =
        unsafe { library.get(b"file_test_stat")? };
    // The file is immutable during every successful mapping's lifetime.
    let result = unsafe { MemOwn::from_file(file.as_file().try_clone()?, dialect) };
    if failure.is_empty() {
        let memory = result?;
        assert_eq!(memory.byte_len(), bytes.len() as u64);
        // Only the mock exposes host-addressable pointers on both strategies.
        assert_eq!(
            unsafe { std::slice::from_raw_parts(memory.as_ptr().cast::<u8>(), bytes.len()) },
            bytes
        );
        let view = memory.view(123, 456)?;
        assert_eq!(view.as_ptr(), memory.as_ptr().wrapping_byte_add(123));
        assert_eq!(
            view.view(12, 20)?.as_ptr(),
            memory.as_ptr().wrapping_byte_add(135)
        );
        assert!(view.view(455, 2).is_err());
        assert!(memory.view(u64::MAX, 1).is_err());
        assert!(memory.view(memory.byte_len(), 0).is_ok());
        drop(memory);
        assert_eq!(unsafe { stat(0) }, usize::from(!integrated));
        assert_eq!(unsafe { stat(1) }, usize::from(!integrated));
        assert_eq!(unsafe { stat(2) }, usize::from(integrated));
        assert_eq!(unsafe { stat(3) }, usize::from(integrated));
        assert_eq!(unsafe { stat(4) }, if integrated { 0 } else { 4 });
        assert!(unsafe { stat(5) } <= 1024 * 1024);
        let empty = tempfile::tempfile()?;
        let memory = unsafe { MemOwn::from_file(empty, dialect)? };
        assert_eq!(memory.byte_len(), 0);
        assert!(memory.as_ptr().is_null());
        assert!(memory.view(0, 0).is_ok());
        assert!(memory.view(0, 1).is_err());
    } else {
        assert!(result.is_err(), "fault injection must fail");
        assert_eq!(
            unsafe { stat(1) },
            usize::from(matches!(failure.as_str(), "copy" | "truncate"))
        );
        assert_eq!(
            unsafe { stat(3) },
            usize::from(matches!(failure.as_str(), "pointer" | "null"))
        );
    }
    Ok(())
}
