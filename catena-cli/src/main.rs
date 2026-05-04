mod compile_graph_render;
mod hexpr_render;

use std::path::PathBuf;

use catena::compile::{
    ArrowType, CompileCheckReport, GraphTheory, check_compile_set, compile_graph,
};
use clap::{Parser, Subcommand, ValueEnum};
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

        #[arg(long, value_enum)]
        theory: CompileTheoryArg,

        #[arg()]
        definition: String,

        /// Write SVG to a file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CompileTheoryArg {
    Data,
    Control,
}

impl From<CompileTheoryArg> for GraphTheory {
    fn from(value: CompileTheoryArg) -> Self {
        match value {
            CompileTheoryArg::Data => GraphTheory::Data,
            CompileTheoryArg::Control => GraphTheory::Control,
        }
    }
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
        } => compile_graph_command(path, theory.into(), &definition, output),
    }
}

fn compile_check_command(path: PathBuf, verbose: bool) -> anyhow::Result<()> {
    let path_display = path.display().to_string();
    let theory_set = TheorySet::from_file(path)?;
    let report = check_compile_set(&theory_set)?;

    print_compile_check_report(&path_display, &report, verbose);
    Ok(())
}

fn compile_graph_command(
    path: PathBuf,
    theory: GraphTheory,
    definition: &str,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let theory_set = TheorySet::from_file(path)?;
    let graph = compile_graph(&theory_set, theory, definition)?;
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

fn print_compile_check_report(path: &str, report: &CompileCheckReport, verbose: bool) {
    println!("OK: compile check passed");
    println!("  file: {path}");
    println!("  data: {} definitions", report.data.definitions_checked);
    println!(
        "  control + lifted data: {} definitions",
        report.control_with_data.definitions_checked
    );
    println!(
        "  data + lifted control: {} definitions",
        report.data_with_control.definitions_checked
    );
    println!(
        "  lifted data -> control: {} arrows",
        report.data_to_control.len()
    );
    println!(
        "  lifted control -> data: {} arrows",
        report.control_to_data.len()
    );

    if verbose {
        print_lift_report("data -> control", &report.data_to_control);
        print_lift_report("control -> data", &report.control_to_data);
    }
}

fn print_lift_report(label: &str, operations: &[ArrowType]) {
    println!("  {label}:");
    for arrow_type in operations {
        println!("    {}", hexpr_render::render_arrow_declaration(arrow_type));
    }
}
