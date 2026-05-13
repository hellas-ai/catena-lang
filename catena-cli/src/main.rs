use std::path::PathBuf;

use catena::compile::{
    CompilePipeline, CompileRequest, Emit, GraphCompileOptions, OutputFormat, compile,
};
use clap::{Parser, Subcommand, ValueEnum};

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
    let mut pipeline = CompilePipeline::new(CompileRequest {
        paths: paths.clone(),
        emit: Emit::Checked,
        theory: None,
        entry: None,
        format: None,
        graph_options: GraphCompileOptions::default(),
    });
    let theory_set = pipeline.checked_elaborated_theory()?;

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
    let generated = compile(CompileRequest {
        paths,
        emit: Emit::Elaborated,
        theory: None,
        entry: None,
        format: None,
        graph_options: GraphCompileOptions::default(),
    })?;
    write_output(None, &generated)
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
    let generated = compile(CompileRequest {
        paths,
        emit: emit.into(),
        theory,
        entry,
        format: format.map(Into::into),
        graph_options: GraphCompileOptions { no_inline },
    })?;

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

impl From<EmitArg> for Emit {
    fn from(value: EmitArg) -> Self {
        match value {
            EmitArg::Cuda => Emit::Cuda,
            EmitArg::CompileGraph => Emit::CompileGraph,
            EmitArg::Elaborated => Emit::Elaborated,
            EmitArg::Checked => Emit::Checked,
            EmitArg::StructuredIr => Emit::StructuredIr,
        }
    }
}

impl From<OutputFormatArg> for OutputFormat {
    fn from(value: OutputFormatArg) -> Self {
        match value {
            OutputFormatArg::Svg => OutputFormat::Svg,
            OutputFormatArg::Text => OutputFormat::Text,
        }
    }
}
