use std::{fs, io, path::Path};

use crate::{codegen::c::render_program, report::CompileReport};

pub fn dump_c(report: &CompileReport, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;

    for (theory_id, programs) in &report.structured_programs {
        for (definition_name, program) in programs {
            let term = report
                .forgotten_closures
                .get(theory_id)
                .and_then(|defs| defs.get(definition_name))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "missing transformed term for `{theory_id}.{definition_name}`"
                        ),
                    )
                })?;
            let rendered = render_program(program, term).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("failed to render C for `{theory_id}.{definition_name}`: {error}"),
                )
            })?;
            fs::write(
                dir.join(format!("{theory_id}.{definition_name}.c")),
                rendered,
            )?;
        }
    }

    Ok(())
}
