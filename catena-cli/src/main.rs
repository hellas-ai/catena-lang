mod compile_graph_render;

use std::path::PathBuf;

use catena::{
    check::check as check_elaborated,
    compile::{CompileConfig, GraphCompileOptions, compile_graph_with_options},
    elaborate::elaborate,
};
use clap::{Parser, Subcommand};
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
        #[arg()]
        path: PathBuf,
    },

    /// Elaborate and typecheck a multi-theory hex file
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

        /// Do not inline definitions matching this pattern. Supports `*`.
        #[arg(long = "no-inline")]
        no_inline: Vec<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Elaborate { path } => elaborate_command(path),
        Command::Check { path, verbose } => check_command(path, verbose),
        Command::Compile { command } => compile_command(command),
    }
}

fn compile_command(command: CompileCommand) -> anyhow::Result<()> {
    match command {
        CompileCommand::Graph {
            path,
            theory,
            definition,
            output,
            no_inline,
        } => compile_graph_command(path, &theory, &definition, output, no_inline),
    }
}

fn check_command(path: PathBuf, verbose: bool) -> anyhow::Result<()> {
    let path_display = path.display().to_string();
    let source = std::fs::read_to_string(path)?;
    let raw = RawTheorySet::from_text(&source)?;
    let elaborated = elaborate(raw)?;
    let theory_set = check_elaborated(&elaborated)?;

    println!("OK: check passed");
    println!("  file: {path_display}");
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

fn elaborate_command(path: PathBuf) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(path)?;
    let raw = RawTheorySet::from_text(&source)?;
    let elaborated = elaborate(raw)?;
    println!("{}", elaborated.to_hexpr_text());
    Ok(())
}

fn compile_graph_command(
    path: PathBuf,
    theory: &str,
    definition: &str,
    output: Option<PathBuf>,
    no_inline: Vec<String>,
) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(path)?;
    let raw = RawTheorySet::from_text(&source)?;
    let elaborated = elaborate(raw)?;
    let config = CompileConfig::data_control();
    let theory_set = check_elaborated(&elaborated)?;
    let graph = compile_graph_with_options(
        &theory_set,
        &config,
        theory,
        definition,
        GraphCompileOptions { no_inline },
    )?;
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
