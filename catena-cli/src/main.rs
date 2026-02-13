use anyhow::anyhow;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use hexpr::*;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};

use catena::pass::{bend::Bend, erase::Erase, unbound::Unbound};

#[derive(Parser)]
#[command(name = "catena", version=env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Erase {
        #[arg()]
        path: PathBuf,
        definition: String,
    },

    Bend {
        #[arg()]
        path: PathBuf,
        definition: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Erase { path, definition } => erase(path, &definition),
        Command::Bend { path, definition } => bend(path, &definition),
    }
}

fn erase(path: PathBuf, definition: &str) -> anyhow::Result<()> {
    let TheoryBundle {
        arrow_theory,
        object_theory,
        definitions,
    } = TheoryBundle::from_file(path)?;

    // NOTE: unwrap() is symptom of bad metacat API design; fix
    let declaration = definitions
        .get(&definition.parse()?)
        .ok_or_else(|| anyhow!("no such definition: {definition}"))?;

    let definition_hexpr = declaration
        .definition
        .clone()
        .ok_or(anyhow!("not a definition: {definition}"))?;

    let mut term = forget_labels(try_interpret(&arrow_theory, &definition_hexpr)?);
    let source = forget_labels(try_interpret(&object_theory, &declaration.source_map)?);
    let target = forget_labels(try_interpret(&object_theory, &declaration.target_map)?);

    let result = check(&arrow_theory, source, target, &mut term)
        .map_err(|e| anyhow!("typechecking failed: {e:?}"))?;

    let term = term.with_nodes(|_| result).unwrap();
    let mut erased = Erase.map_arrow(&term);
    erased.quotient();

    // Tell pretty-printer the coarity of each operation
    let coarity = |op: &OperationKey| -> usize { object_theory.type_maps(op).1.targets.len() };

    // Pretty-print node labels using computed types
    let labels: Vec<String> = erased
        .hypergraph
        .nodes
        .iter()
        .map(|n| n.pretty(Some(&coarity)))
        .collect();

    use open_hypergraphs_dot::{Options, svg::to_svg_with};
    use std::io::Write;

    let opts = Options::default().display();
    std::io::stdout().write_all(&to_svg_with(
        &erased
            .with_nodes(|_| labels)
            .expect("labels length mismatch"),
        &opts,
    )?)?;

    Ok(())
}

fn bend(path: PathBuf, definition: &str) -> anyhow::Result<()> {
    let TheoryBundle {
        arrow_theory,
        object_theory,
        definitions,
    } = TheoryBundle::from_file(path)?;

    // NOTE: unwrap() is symptom of bad metacat API design; fix
    let declaration = definitions
        .get(&definition.parse()?)
        .ok_or_else(|| anyhow!("no such definition: {definition}"))?;

    let definition_hexpr = declaration
        .definition
        .clone()
        .ok_or(anyhow!("not a definition: {definition}"))?;

    let mut term = forget_labels(try_interpret(&arrow_theory, &definition_hexpr)?);
    let source = forget_labels(try_interpret(&object_theory, &declaration.source_map)?);
    let target = forget_labels(try_interpret(&object_theory, &declaration.target_map)?);

    let result = check(&arrow_theory, source, target, &mut term)
        .map_err(|e| anyhow!("typechecking failed: {e:?}"))?;

    let term = term.with_nodes(|_| result).unwrap();

    // Erase
    let mut erased = Erase.map_arrow(&term);
    erased.quotient();

    // bound → value
    let mut unbound = Unbound.map_arrow(&erased);
    unbound.quotient();

    let mut bent = Bend.map_arrow(&unbound);
    bent.quotient();

    let term = bent;

    // Tell pretty-printer the coarity of each operation
    let coarity = |op: &OperationKey| -> usize { object_theory.type_maps(op).1.targets.len() };

    // Pretty-print node labels using computed types
    let labels: Vec<String> = term
        .hypergraph
        .nodes
        .iter()
        .map(|n| n.pretty(Some(&coarity)))
        .collect();

    use open_hypergraphs_dot::{Options, svg::to_svg_with};
    use std::io::Write;

    let opts = Options::default().display();
    std::io::stdout().write_all(&to_svg_with(
        &term.with_nodes(|_| labels).expect("labels length mismatch"),
        &opts,
    )?)?;

    Ok(())
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
