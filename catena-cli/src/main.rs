mod compile_check_report;
mod compile_graph_render;
mod hexpr_render;

use std::path::PathBuf;

use catena::compile::{
    CompileConfig, check_compile_theories, compile_graph, load_extended_theory_set_from_text,
};
use clap::{Parser, Subcommand};

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
    let source = std::fs::read_to_string(path)?;
    let config = CompileConfig::data_control();
    let theory_set = load_extended_theory_set_from_text(&source, &config)?;
    let report = check_compile_theories(&theory_set, &config)?;

    compile_check_report::print_compile_check_report(&path_display, &report, verbose);
    Ok(())
}

fn compile_graph_command(
    path: PathBuf,
    theory: &str,
    definition: &str,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(path)?;
    let config = CompileConfig::data_control();
    let theory_set = load_extended_theory_set_from_text(&source, &config)?;
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
