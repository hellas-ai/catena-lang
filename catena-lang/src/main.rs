use std::{fs, path::PathBuf};

use clap::Parser;
use metacat::theory::RawTheorySet;

#[derive(Parser)]
#[command(name = "catena-dsl", version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    #[arg(short, long)]
    output_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let sources = cli
        .paths
        .iter()
        .map(fs::read_to_string)
        .collect::<Result<Vec<_>, _>>()?;
    let mut all_sources: Vec<&str> = catena_lang::stdlib::sources().collect();
    all_sources.extend(sources.iter().map(String::as_str));
    let raw_theories = RawTheorySet::from_texts(all_sources)?;
    match catena_lang::compile::compile(raw_theories) {
        Ok(report) => {
            report.dump_to_dir(&cli.output_dir)?;
            Ok(())
        }
        Err(failure) => {
            failure.report.dump_to_dir(&cli.output_dir)?;
            Err(failure.into())
        }
    }
}
