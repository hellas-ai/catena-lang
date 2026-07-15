use std::{env, path::PathBuf};

use catena_lang::{
    compile::{CompileError, compile},
    stdlib,
};
use metacat::theory::RawTheorySet;

fn main() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(stdlib::sources().chain([include_str!("closure2.hex")]))?;
    let report = match compile(raw) {
        Ok(report) => report,
        Err(failure) if matches!(failure.cause, CompileError::NotImplementedError) => {
            failure.report
        }
        Err(failure) => return Err(failure.into()),
    };

    let output = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/region-report")
    });
    report.dump_to_dir(&output)?;
    println!("wrote region report to {}", output.display());
    Ok(())
}
