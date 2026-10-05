use std::{fs, path::PathBuf};

use catena_lang::{codegen::CodegenKind, report::ReportOptions};
use clap::{Parser, ValueEnum};
use metacat::theory::RawTheorySet;

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    #[arg(short, long)]
    output_dir: PathBuf,

    /// Skip rendering compiler graphs as SVG files.
    #[arg(long)]
    no_svg: bool,

    /// Code generator to use.
    #[arg(long, value_enum, default_value_t)]
    codegen: CodegenArg,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum CodegenArg {
    /// The existing GPU C++ code generator.
    #[default]
    Gpu,
}

impl From<CodegenArg> for CodegenKind {
    fn from(value: CodegenArg) -> Self {
        match value {
            CodegenArg::Gpu => Self::Gpu,
        }
    }
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
    let report_options = ReportOptions {
        #[cfg(feature = "svg-reports")]
        generate_svgs: !cli.no_svg,
    };
    match catena_lang::compile::compile_with_codegen(raw_theories, cli.codegen.into()) {
        Ok(report) => {
            report.dump_to_dir_with_options(&cli.output_dir, report_options)?;
            Ok(())
        }
        Err(failure) => {
            failure
                .report
                .dump_to_dir_with_options(&cli.output_dir, report_options)?;
            Err(failure.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_generation_is_enabled_by_default() {
        let cli = Cli::try_parse_from(["catena", "input.hex", "--output-dir", "report"]).unwrap();

        assert!(!cli.no_svg);
    }

    #[test]
    fn no_svg_flag_disables_svg_generation() {
        let cli =
            Cli::try_parse_from(["catena", "input.hex", "--output-dir", "report", "--no-svg"])
                .unwrap();

        assert!(cli.no_svg);
    }

    #[test]
    fn gpu_codegen_is_the_default() {
        let cli = Cli::try_parse_from(["catena", "input.hex", "--output-dir", "report"]).unwrap();

        assert!(matches!(cli.codegen, CodegenArg::Gpu));
    }

    #[test]
    fn gpu_codegen_can_be_selected_explicitly() {
        let cli = Cli::try_parse_from([
            "catena",
            "input.hex",
            "--output-dir",
            "report",
            "--codegen",
            "gpu",
        ])
        .unwrap();

        assert!(matches!(cli.codegen, CodegenArg::Gpu));
    }
}
