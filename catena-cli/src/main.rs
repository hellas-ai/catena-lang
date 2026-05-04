mod compile_check_report;
mod compile_graph_render;
mod hexpr_render;

use std::path::PathBuf;

use catena::compile::{CompileConfig, check_compile_set, compile_graph};
use clap::{Parser, Subcommand};
use metacat::theory::TheorySet;

#[derive(Parser)]
#[command(name = "catena", version = env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check a multi-theory hex file with metacat/Catena compile checks
    Check {
        #[arg()]
        path: PathBuf,

        #[arg(long)]
        verbose: bool,
    },

    /// Run the Catena compile pipeline
    Compile {
        #[command(subcommand)]
        command: CompileCommand,
    },
}

#[derive(Subcommand)]
enum CompileCommand {
    /// Check data/control theories after Catena lift passes
    Check {
        #[arg()]
        path: PathBuf,

        #[arg(long)]
        verbose: bool,
    },

    /// Render one compile graph as SVG, inlining only same-theory definitions
    Graph {
        #[arg()]
        path: PathBuf,

        #[arg(long)]
        theory: String,

        #[arg()]
        definition: String,

        /// Write SVG to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Check { path, verbose } => compile_check_command(path, verbose),
        Command::Compile { command } => compile_command(command),
    }
}

fn compile_command(command: CompileCommand) -> anyhow::Result<()> {
    match command {
        CompileCommand::Check { path, verbose } => compile_check_command(path, verbose),
        CompileCommand::Graph {
            path,
            theory,
            definition,
            output,
        } => compile_graph_command(path, &theory, &definition, output),
    }
}

fn compile_check_command(path: PathBuf, verbose: bool) -> anyhow::Result<()> {
    let path_display = path.display().to_string();
    let theory_set = TheorySet::from_file(path)?;
    let report = check_compile_set(&theory_set)?;

    compile_check_report::print_compile_check_report(&path_display, &report, verbose);
    Ok(())
}

fn compile_graph_command(
    path: PathBuf,
    theory: &str,
    definition: &str,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let theory_set = TheorySet::from_file(path)?;
    let config = CompileConfig::data_control();
    let graph = compile_graph(&theory_set, &config, theory, definition)?;
    let svg = compile_graph_render::nested_svg(&graph)?;

    match output {
        Some(output) => std::fs::write(output, svg)?,
        None => {
            use std::io::Write;
            std::io::stdout().write_all(&svg)?;
        }
    }

    Ok(())
}
