use std::{fs, path::PathBuf};

use catena_lang::report::DumpOptions;
use clap::Parser;
use metacat::theory::RawTheorySet;

#[derive(Parser)]
#[command(name = "catena-dsl", version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    #[arg(short, long)]
    output_dir: PathBuf,

    /// Skip rendering compiler graphs as SVG files.
    #[arg(long)]
    no_svg: bool,
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
    let dump_options = DumpOptions {
        generate_svgs: !cli.no_svg,
    };
    match catena_lang::compile::compile(raw_theories) {
        Ok(report) => {
            report.dump_to_dir_with_options(&cli.output_dir, dump_options)?;
            Ok(())
        }
        Err(failure) => {
            failure
                .report
                .dump_to_dir_with_options(&cli.output_dir, dump_options)?;
            Err(failure.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_generation_is_enabled_by_default() {
        let cli =
            Cli::try_parse_from(["catena-dsl", "input.hex", "--output-dir", "report"]).unwrap();

        assert!(!cli.no_svg);
    }

    #[test]
    fn no_svg_flag_disables_svg_generation() {
        let cli = Cli::try_parse_from([
            "catena-dsl",
            "input.hex",
            "--output-dir",
            "report",
            "--no-svg",
        ])
        .unwrap();

        assert!(cli.no_svg);
    }
}
