pub mod check;
pub mod codegen;
pub mod compile;
pub mod elaborate;
pub mod pass;
pub mod report;

use std::path::PathBuf;

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
    let raw_theories = RawTheorySet::from_files(cli.paths)?;
    let report = compile::compile(raw_theories)?;
    report.dump_to_dir(cli.output_dir)?;
    Ok(())
}
