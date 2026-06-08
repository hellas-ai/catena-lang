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
    match catena_dsl::compile::compile(raw_theories) {
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
