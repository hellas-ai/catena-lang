use std::{fs::File, io::Write};

use catena_lang::safe_gpu::{AssetError, GpuDialect, Session, run_worker_if_requested};

fn main() -> anyhow::Result<()> {
    if run_worker_if_requested()? {
        return Ok(());
    }

    let session = Session::new(GpuDialect::Hip)?;
    let key = [0x42; 32];
    let asset = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(asset.byte_len() == 4096);
    anyhow::ensure!(asset == asset.clone());

    let slice = session.slice(&asset, 128, 1024)?;
    anyhow::ensure!(slice.offset() == 128);
    anyhow::ensure!(slice.byte_len() == 1024);
    anyhow::ensure!(session.slice(&asset, 4096, 0).is_ok());
    anyhow::ensure!(matches!(
        session.slice(&asset, 4090, 8),
        Err(AssetError::InvalidSlice { .. })
    ));
    anyhow::ensure!(matches!(
        session.slice(&asset, u64::MAX, 2),
        Err(AssetError::InvalidSlice { .. })
    ));

    // A stable opaque handle proves the resident mapping was reused rather
    // than registered again for the same content identity.
    let reused = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(reused == asset);

    let conflicting = session.attach(key, unlinked_read_only_file(&[7; 8192])?);
    anyhow::ensure!(matches!(conflicting, Err(AssetError::Remote(_))));

    let empty = session.attach([0x43; 32], unlinked_read_only_file(&[])?);
    anyhow::ensure!(matches!(empty, Err(AssetError::InvalidLength { .. })));
    let writable = tempfile::tempfile()?;
    writable.set_len(4096)?;
    anyhow::ensure!(matches!(
        session.attach([0x44; 32], writable),
        Err(AssetError::Remote(_))
    ));

    let other_session = Session::new(GpuDialect::Hip)?;
    anyhow::ensure!(matches!(
        other_session.slice(&asset, 0, 1),
        Err(AssetError::WrongSession)
    ));
    Ok(())
}

fn unlinked_read_only_file(bytes: &[u8]) -> anyhow::Result<File> {
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    let path = temporary.path().to_owned();
    let file = File::open(&path)?;
    temporary.close()?;
    anyhow::ensure!(!path.exists(), "temporary asset path still exists");
    Ok(file)
}
