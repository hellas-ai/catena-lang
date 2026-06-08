use std::{fs, io, path::Path};

use crate::{codegen::gpu::render_modules, report::CompileReport};

pub fn dump_gpu(report: &CompileReport, dir: &Path) -> io::Result<()> {
    let Some(gpu_modules) = &report.gpu_modules else {
        return Ok(());
    };

    fs::create_dir_all(dir)?;
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "cpp") {
            fs::remove_file(path)?;
        }
    }

    let rendered = render_modules(gpu_modules).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to render GPU code: {error}"),
        )
    })?;
    fs::write(dir.join("program.cpp"), rendered)?;

    Ok(())
}
