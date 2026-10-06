use std::{fs, path::PathBuf};

use anyhow::Context;
use catena_lang::{
    codegen::CodegenKind,
    compile::{ProgressEvent, StageStatus},
    report::{CompileReport, ReportOptions},
    runtime::GpuDialect,
    stdlib::{BundleRegistry, SourceFile},
};
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

    /// GPU dialect to generate.
    #[arg(long, value_enum, default_value_t)]
    dialect: DialectArg,

    /// Select a named stdlib bundle (repeatable). Replaces the implicit default.
    #[arg(long = "stdlib", value_name = "NAME")]
    stdlibs: Vec<String>,

    /// Load a local bundle's stdlib.json (repeatable). Replaces the implicit default.
    #[arg(long = "stdlib-dir", value_name = "PATH")]
    stdlib_dirs: Vec<PathBuf>,

    /// Compile only the supplied input files, without a standard library.
    #[arg(long, conflicts_with_all = ["stdlibs", "stdlib_dirs"])]
    no_stdlib: bool,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum CodegenArg {
    /// The default GPU C++ code generator.
    #[default]
    Default,
    /// The standalone experimental GPU C++ code generator.
    Experimental,
}

impl From<CodegenArg> for CodegenKind {
    fn from(value: CodegenArg) -> Self {
        match value {
            CodegenArg::Default => Self::Default,
            CodegenArg::Experimental => Self::Experimental,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum DialectArg {
    #[default]
    Hip,
    Cuda,
}

impl From<DialectArg> for GpuDialect {
    fn from(value: DialectArg) -> Self {
        match value {
            DialectArg::Hip => Self::Hip,
            DialectArg::Cuda => Self::Cuda,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let raw_theories = load_theories(&cli)?;
    let report_options = ReportOptions {
        #[cfg(feature = "svg-reports")]
        generate_svgs: !cli.no_svg,
    };
    let mut report = CompileReport::new(raw_theories);
    let result = catena_lang::compile::compile_with_progress(
        &mut report,
        cli.codegen.into(),
        cli.dialect.into(),
        &mut print_progress,
    );
    report.dump_graphs_to_dir_with_progress(
        &cli.output_dir,
        report_options,
        &mut print_progress,
    )?;
    let module = result?;
    let dir = cli.output_dir.join("gpu");
    fs::create_dir_all(&dir)?;
    let filename = match module.dialect {
        GpuDialect::Hip => "hip.cpp",
        GpuDialect::Cuda => "cuda.cpp",
    };
    fs::write(dir.join(filename), module.source)?;
    Ok(())
}

fn print_progress(event: ProgressEvent) {
    match event {
        ProgressEvent::Started { stage } => eprintln!("[compile] {stage} started"),
        ProgressEvent::Update {
            stage,
            message,
            elapsed_ms,
        } => {
            eprintln!(
                "[compile] {stage}: {message} — elapsed {:.2}s",
                elapsed_ms / 1000.0
            );
        }
        ProgressEvent::Finished(timing) => {
            let status = match timing.status {
                StageStatus::Completed => "completed",
                StageStatus::Failed => "failed",
            };
            eprintln!(
                "[compile] {} {status} in {:.2}s",
                timing.stage,
                timing.elapsed_ms / 1000.0
            );
        }
    }
}

fn selected_stdlib(cli: &Cli) -> anyhow::Result<Vec<SourceFile>> {
    if cli.no_stdlib {
        return Ok(Vec::new());
    }
    let mut registry = BundleRegistry::default();
    let mut names = cli.stdlibs.clone();
    for directory in &cli.stdlib_dirs {
        names.push(registry.add_directory(directory)?);
    }
    if names.is_empty() {
        names.push("default".into());
    }
    registry.resolve(&names.iter().map(String::as_str).collect::<Vec<_>>())
}

fn load_theories(cli: &Cli) -> anyhow::Result<RawTheorySet> {
    let mut sources = selected_stdlib(cli)?;
    for filename in &cli.paths {
        let source = fs::read_to_string(filename)
            .with_context(|| format!("failed to read input {}", filename.display()))?;
        sources.push(SourceFile {
            filename: filename.clone(),
            source: source.into(),
        });
    }
    let mut theories = RawTheorySet::from_texts(std::iter::empty::<&str>())?;
    for file in sources {
        let parsed = RawTheorySet::from_text(&file.source)
            .with_context(|| format!("failed to parse {}", file.filename.display()))?;
        theories = theories
            .merge(parsed)
            .with_context(|| format!("failed to merge {}", file.filename.display()))?;
    }
    Ok(theories)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli_with(flags: &[&str]) -> Cli {
        Cli::try_parse_from(
            ["catena", "input.hex", "-o", "report"]
                .into_iter()
                .chain(flags.iter().copied()),
        )
        .unwrap()
    }

    #[test]
    fn stdlib_default_explicit_and_disabled_selection() -> anyhow::Result<()> {
        let default = selected_stdlib(&cli_with(&[]))?;
        let explicit = selected_stdlib(&cli_with(&["--stdlib", "default", "--stdlib", "default"]))?;
        assert_eq!(
            default.iter().map(|file| &file.source).collect::<Vec<_>>(),
            explicit.iter().map(|file| &file.source).collect::<Vec<_>>()
        );
        assert!(!default.is_empty());
        assert!(selected_stdlib(&cli_with(&["--no-stdlib"]))?.is_empty());
        assert!(selected_stdlib(&cli_with(&["--stdlib", "missing"])).is_err());
        for flag in ["--stdlib", "--stdlib-dir"] {
            assert!(
                Cli::try_parse_from([
                    "catena",
                    "input.hex",
                    "-o",
                    "report",
                    "--no-stdlib",
                    flag,
                    "default"
                ])
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn local_bundles_replace_default_and_can_depend_on_each_other() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let base = directory.path().join("base");
        let extension = directory.path().join("extension");
        fs::create_dir(&base)?;
        fs::create_dir(&extension)?;
        fs::write(
            base.join("stdlib.json"),
            r#"{"name":"base","files":["base.hex"]}"#,
        )?;
        fs::write(base.join("base.hex"), "# base")?;
        fs::write(
            extension.join("stdlib.json"),
            r#"{"name":"extension","extends":["base"],"files":["extension.hex"]}"#,
        )?;
        fs::write(extension.join("extension.hex"), "# extension")?;
        let cli = cli_with(&[
            "--stdlib",
            "extension",
            "--stdlib-dir",
            extension.to_str().unwrap(),
            "--stdlib-dir",
            base.to_str().unwrap(),
        ]);
        let files = selected_stdlib(&cli)?;
        assert_eq!(
            files
                .iter()
                .map(|file| file.source.as_ref())
                .collect::<Vec<_>>(),
            ["# base", "# extension"]
        );
        Ok(())
    }

    #[test]
    fn local_extension_compiles_and_duplicate_definitions_fail() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        fs::write(
            directory.path().join("stdlib.json"),
            r#"{"name":"custom","extends":["default"],"files":["custom.hex"]}"#,
        )?;
        let definition = "(def program custom.one : [] -> (u64 val) = (u64.one))";
        fs::write(directory.path().join("custom.hex"), definition)?;
        let input = directory.path().join("program.hex");
        fs::write(
            &input,
            "(def program main : [] -> (u64 val) = (custom.one))",
        )?;
        let mut cli = cli_with(&["--stdlib-dir", directory.path().to_str().unwrap()]);
        cli.paths = vec![input.clone()];
        catena_lang::compile::compile(
            &mut CompileReport::new(load_theories(&cli)?),
            cli.codegen.into(),
            cli.dialect.into(),
        )?;

        fs::write(&input, definition)?;
        let theories = load_theories(&cli)?;
        let error = catena_lang::compile::compile(
            &mut CompileReport::new(theories),
            cli.codegen.into(),
            cli.dialect.into(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("custom.one multiple times"), "{error}");
        Ok(())
    }

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
    fn default_codegen_is_the_default() {
        let cli = Cli::try_parse_from(["catena", "input.hex", "--output-dir", "report"]).unwrap();

        assert!(matches!(cli.codegen, CodegenArg::Default));
    }

    #[test]
    fn default_codegen_can_be_selected_explicitly() {
        let cli = Cli::try_parse_from([
            "catena",
            "input.hex",
            "--output-dir",
            "report",
            "--codegen",
            "default",
        ])
        .unwrap();

        assert!(matches!(cli.codegen, CodegenArg::Default));
    }

    #[test]
    fn experimental_codegen_is_explicit_and_does_not_select_a_stdlib() {
        let cli = cli_with(&["--codegen", "experimental"]);
        assert_eq!(CodegenKind::from(cli.codegen), CodegenKind::Experimental);
        assert!(cli.stdlibs.is_empty());
        assert!(cli.stdlib_dirs.is_empty());
    }

    #[test]
    fn gpu_is_not_a_codegen_alias() {
        assert!(
            Cli::try_parse_from(["catena", "input.hex", "-o", "report", "--codegen", "gpu"])
                .is_err()
        );
    }
}
