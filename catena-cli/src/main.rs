mod compile_graph_render;

use std::path::PathBuf;

use catena::{
    check::check as check_elaborated,
    compile::{
        CompileConfig, GraphCompileOptions, compile_graph_with_options,
        cuda::{CudaEmit, compile_cuda_theory_set_with_options},
    },
    elaborate::elaborate,
};
use clap::{Parser, Subcommand, ValueEnum};
use metacat::theory::RawTheorySet;

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Elaborate a multi-theory hex file by interleaving control/data theories
    Elaborate {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },

    /// Elaborate and typecheck a multi-theory hex file
    Check {
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[arg(long)]
        verbose: bool,
    },

    /// Run the Catena compile pipeline
    Compile {
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        #[arg(long)]
        emit: EmitArg,

        #[arg(long)]
        theory: Option<String>,

        #[arg(long)]
        entry: Option<String>,

        #[arg(long, value_enum)]
        format: Option<OutputFormatArg>,

        /// Write output to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Do not inline definitions matching this pattern. Supports `*`.
        #[arg(long = "no-inline")]
        no_inline: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EmitArg {
    Cuda,
    CompileGraph,
    Elaborated,
    Checked,
    StructuredIr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormatArg {
    Svg,
    Text,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Elaborate { paths } => elaborate_command(paths),
        Command::Check { paths, verbose } => check_command(paths, verbose),
        Command::Compile {
            paths,
            emit,
            theory,
            entry,
            format,
            output,
            no_inline,
        } => compile_command(paths, emit, theory, entry, format, output, no_inline),
    }
}

fn check_command(paths: Vec<PathBuf>, verbose: bool) -> anyhow::Result<()> {
    let raw = RawTheorySet::from_files(paths.clone())?;
    let elaborated = elaborate(raw)?;
    let theory_set = check_elaborated(&elaborated)?;

    println!("OK: check passed");
    if paths.len() == 1 {
        println!("  file: {}", paths[0].display());
    } else {
        println!("  files: {}", paths.len());
    }
    if verbose {
        for (id, theory) in &theory_set.theories {
            if let metacat::theory::Theory::Theory { arrows, .. } = theory {
                let definitions = arrows
                    .values()
                    .filter(|arrow| arrow.definition.is_some())
                    .count();
                println!("  {}: {} definitions", id, definitions);
            }
        }
    }
    Ok(())
}

fn elaborate_command(paths: Vec<PathBuf>) -> anyhow::Result<()> {
    let elaborated = load_elaborated(paths)?;
    println!("{}", elaborated.to_hexpr_text());
    Ok(())
}

fn compile_command(
    paths: Vec<PathBuf>,
    emit: EmitArg,
    theory: Option<String>,
    entry: Option<String>,
    format: Option<OutputFormatArg>,
    output: Option<PathBuf>,
    no_inline: Vec<String>,
) -> anyhow::Result<()> {
    let generated = match emit {
        EmitArg::Elaborated => {
            require_format(format, OutputFormatArg::Text, emit)?;
            reject_no_inline(&no_inline, emit)?;
            load_elaborated(paths)?.to_hexpr_text().into_bytes()
        }
        EmitArg::Checked => {
            require_format(format, OutputFormatArg::Text, emit)?;
            reject_no_inline(&no_inline, emit)?;
            let theory_set = load_checked(paths)?;
            check_summary(&theory_set).into_bytes()
        }
        EmitArg::CompileGraph => {
            require_format(format, OutputFormatArg::Svg, emit)?;
            let theory = required_arg(theory, "--theory", emit)?;
            let entry = required_arg(entry, "--entry", emit)?;
            let theory_set = load_checked(paths)?;
            let graph = compile_graph_with_options(
                &theory_set,
                &CompileConfig::data_control(),
                &theory,
                &entry,
                GraphCompileOptions { no_inline },
            )?;
            compile_graph_render::nested_svg(&graph)?
        }
        EmitArg::Cuda | EmitArg::StructuredIr => {
            require_format(format, OutputFormatArg::Text, emit)?;
            let theory = required_arg(theory, "--theory", emit)?;
            let entry = required_arg(entry, "--entry", emit)?;
            let theory_set = load_checked(paths)?;
            compile_cuda_theory_set_with_options(
                &theory_set,
                &theory,
                &entry,
                emit.into(),
                GraphCompileOptions { no_inline },
            )?
            .into_bytes()
        }
    };

    write_output(output, &generated)
}

fn write_output(output: Option<PathBuf>, generated: &[u8]) -> anyhow::Result<()> {
    match output {
        Some(output) => std::fs::write(output, generated)?,
        None => {
            use std::io::Write;
            std::io::stdout().write_all(generated)?;
        }
    }

    Ok(())
}

fn load_elaborated(paths: Vec<PathBuf>) -> anyhow::Result<RawTheorySet> {
    let raw = RawTheorySet::from_files(paths)?;
    Ok(elaborate(raw)?)
}

fn load_checked(paths: Vec<PathBuf>) -> anyhow::Result<metacat::theory::TheorySet> {
    let elaborated = load_elaborated(paths)?;
    Ok(check_elaborated(&elaborated)?)
}

fn check_summary(theory_set: &metacat::theory::TheorySet) -> String {
    let mut lines = vec!["OK: check passed".to_string()];
    for (id, theory) in &theory_set.theories {
        if let metacat::theory::Theory::Theory { arrows, .. } = theory {
            let definitions = arrows
                .values()
                .filter(|arrow| arrow.definition.is_some())
                .count();
            lines.push(format!("  {id}: {definitions} definitions"));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn required_arg(value: Option<String>, name: &str, emit: EmitArg) -> anyhow::Result<String> {
    value.ok_or_else(|| anyhow::anyhow!("{name} is required when emitting {emit:?}"))
}

fn require_format(
    format: Option<OutputFormatArg>,
    expected: OutputFormatArg,
    emit: EmitArg,
) -> anyhow::Result<()> {
    if let Some(format) = format
        && format != expected
    {
        anyhow::bail!("--format {format:?} is not supported when emitting {emit:?}");
    }
    Ok(())
}

fn reject_no_inline(no_inline: &[String], emit: EmitArg) -> anyhow::Result<()> {
    if !no_inline.is_empty() {
        anyhow::bail!(
            "--no-inline is only supported for emits that build a compile graph, not {emit:?}"
        );
    }
    Ok(())
}

impl From<EmitArg> for CudaEmit {
    fn from(value: EmitArg) -> Self {
        match value {
            EmitArg::Cuda => CudaEmit::Cuda,
            EmitArg::StructuredIr => CudaEmit::StructuredIr,
            _ => unreachable!("only cuda emit variants can be converted to CudaEmit"),
        }
    }
}
