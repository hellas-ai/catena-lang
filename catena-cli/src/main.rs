use catena::lower::{Pass, lower};
use catena::shallow::shallow_graph;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use catena::backend::c::codegen::codegen;
use catena::lang::Obj;
use metacat::{syntax::TheoryBundle, theory::OperationKey};

#[derive(Parser)]
#[command(name = "catena", version=env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run codegen for a given pass
    Codegen {
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },

    /// Run compiler passes up to the given pass and output SVG
    Lower {
        #[arg()]
        pass: PassArg,
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },

    /// Check one definition and output its graph without inlining called arrows
    ShallowGraph {
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PassArg {
    Check,
    Erase,
    ForgetBound,
    ExpandEta,
    DiscardNaturality,
}

impl From<PassArg> for Pass {
    fn from(value: PassArg) -> Self {
        match value {
            PassArg::Check => Pass::Check,
            PassArg::Erase => Pass::Erase,
            PassArg::ForgetBound => Pass::ForgetBound,
            PassArg::ExpandEta => Pass::ExpandEta,
            PassArg::DiscardNaturality => Pass::DiscardNaturality,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Codegen { path, definition } => {
            let bundle = TheoryBundle::from_file(path)?;
            let lowered = lower(&bundle, Pass::DiscardNaturality, &definition)?;
            println!("{}", codegen(lowered, "out"));
            Ok(())
        }
        Command::Lower {
            path,
            pass,
            definition,
        } => lower_command(TheoryBundle::from_file(path)?, pass.into(), &definition),
        Command::ShallowGraph { path, definition } => {
            shallow_graph_command(TheoryBundle::from_file(path)?, &definition)
        }
    }
}

fn lower_command(bundle: TheoryBundle, until: Pass, definition: &str) -> anyhow::Result<()> {
    let current = lower(&bundle, until, definition)?;
    print_svg(&bundle, current)
}

fn shallow_graph_command(bundle: TheoryBundle, definition: &str) -> anyhow::Result<()> {
    let current = shallow_graph(&bundle, definition)?;
    print_svg(&bundle, current)
}

fn print_svg(
    bundle: &TheoryBundle,
    current: open_hypergraphs::lax::OpenHypergraph<Obj, OperationKey>,
) -> anyhow::Result<()> {
    // Pretty-print
    let coarity =
        |op: &OperationKey| -> usize { bundle.object_theory.type_maps(op).1.targets.len() };

    let labels: Vec<String> = current
        .hypergraph
        .nodes
        .iter()
        .map(|n| n.pretty(Some(&coarity)))
        .collect();

    use open_hypergraphs_dot::{Options, svg::to_svg_with};
    use std::io::Write;

    let opts = Options::default().display();
    std::io::stdout().write_all(&to_svg_with(
        &current
            .with_nodes(|_| labels)
            .expect("labels length mismatch"),
        &opts,
    )?)?;

    Ok(())
}
