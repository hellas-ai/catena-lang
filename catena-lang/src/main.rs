use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

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

mod source_merge;

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Input files or directories. Directories load all .hex files recursively.
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

    /// Load all .hex files recursively from a directory (repeatable). Replaces the implicit default.
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
    let registry = BundleRegistry::default();
    let mut names = cli.stdlibs.clone();
    if names.is_empty() && cli.stdlib_dirs.is_empty() {
        names.push("default".into());
    }
    let mut sources = registry.resolve(&names.iter().map(String::as_str).collect::<Vec<_>>())?;
    for filename in input_files(&cli.stdlib_dirs)? {
        sources.push(read_source(filename)?);
    }
    Ok(sources)
}

fn collect_hex_files(directory: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("failed to read input directory {}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        // Do not follow directory symlinks, which can introduce traversal cycles.
        if entry.file_type()?.is_dir() {
            collect_hex_files(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "hex") && path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn input_files(paths: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let start = files.len();
            collect_hex_files(path, &mut files)?;
            anyhow::ensure!(
                files.len() > start,
                "no .hex files found in {}",
                path.display()
            );
        } else {
            files.push(path.clone());
        }
    }
    let mut seen = BTreeSet::new();
    let mut unique = Vec::new();
    for path in files {
        let canonical = fs::canonicalize(&path)
            .with_context(|| format!("failed to resolve input {}", path.display()))?;
        if seen.insert(canonical) {
            unique.push(path);
        }
    }
    Ok(unique)
}

fn read_source(filename: PathBuf) -> anyhow::Result<SourceFile> {
    let source = fs::read_to_string(&filename)
        .with_context(|| format!("failed to read input {}", filename.display()))?;
    Ok(SourceFile {
        filename,
        source: source.into(),
    })
}

fn load_theories(cli: &Cli) -> anyhow::Result<RawTheorySet> {
    let mut sources = selected_stdlib(cli)?;
    for filename in input_files(&cli.paths)? {
        sources.push(read_source(filename)?);
    }
    let mut seen = BTreeSet::new();
    let mut theories = RawTheorySet::from_texts(std::iter::empty::<&str>())?;
    for file in sources {
        let identity = fs::canonicalize(&file.filename).unwrap_or_else(|_| file.filename.clone());
        if !seen.insert(identity) {
            continue;
        }
        let parsed = RawTheorySet::from_text(&file.source)
            .with_context(|| format!("failed to parse {}", file.filename.display()))?;
        theories = source_merge::merge(theories, parsed)
            .with_context(|| format!("failed to merge {}", file.filename.display()))?;
    }
    Ok(theories)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directory_inputs_are_recursive_sorted_and_deduplicated() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        fs::create_dir_all(root.join("nested/deeper"))?;
        let first = root.join("a.hex");
        let second = root.join("nested/deeper/b.hex");
        fs::write(
            &first,
            "(theory program type {(arr first : ([] {}) -> ([] {}))})",
        )?;
        fs::write(
            &second,
            "(theory program type {(arr second : ([] {}) -> ([] {}))})",
        )?;
        fs::write(root.join("ignored.txt"), "invalid hex syntax")?;
        fs::write(root.join("nested/ignored.cpp"), "invalid hex syntax")?;
        let mut cli = cli_with(&["--no-stdlib"]);
        cli.paths = vec![root.to_path_buf(), first.clone(), root.join("nested")];
        assert_eq!(input_files(&cli.paths)?, [first, second]);
        let theories = load_theories(&cli)?;
        let arrows = &theories.theories[&"program".parse()?].arrows;
        assert!(arrows.contains_key(&"first".parse()?));
        assert!(arrows.contains_key(&"second".parse()?));
        Ok(())
    }

    #[test]
    fn empty_input_directory_is_an_error() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let error = input_files(&[directory.path().to_path_buf()]).unwrap_err();
        assert!(error.to_string().contains("no .hex files found"));
        Ok(())
    }

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
    fn local_directories_load_recursively_without_manifests() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let base = directory.path().join("base");
        let extension = directory.path().join("extension");
        fs::create_dir_all(base.join("nested"))?;
        fs::create_dir(&extension)?;
        fs::write(base.join("nested/base.hex"), "# base")?;
        fs::write(extension.join("extension.hex"), "# extension")?;
        let cli = cli_with(&[
            "--stdlib-dir",
            base.to_str().unwrap(),
            "--stdlib-dir",
            extension.to_str().unwrap(),
        ]);
        for manifest in [None, Some("invalid manifest that must be ignored")] {
            if let Some(contents) = manifest {
                fs::write(base.join("stdlib.json"), contents)?;
            }
            let files = selected_stdlib(&cli)?;
            assert_eq!(
                files
                    .iter()
                    .map(|file| file.source.as_ref())
                    .collect::<Vec<_>>(),
                ["# base", "# extension"]
            );
        }
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
        let mut cli = cli_with(&[
            "--stdlib",
            "default",
            "--stdlib-dir",
            directory.path().to_str().unwrap(),
        ]);
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
