use hexpr::{Operation, try_interpret};
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::OpenHypergraph;
use open_hypergraphs::strict::vec::FiniteFunction;
use thiserror::Error;

use crate::lang::{Arr, Obj};

#[derive(Error, Debug)]
pub enum ShallowError {
    #[error("Invalid quotient: {0:?}")]
    InvalidQuotient(FiniteFunction),
    #[error("Unknown definition {0}")]
    UnknownDefinition(String),
    #[error("Invalid definition name: {0}")]
    InvalidDefinition(String),
    #[error("Invalid hexpr: {0}")]
    InvalidHexpr(#[from] hexpr::interpret::Error<metacat::theory::Error>),
    #[error("Typecheck failed: {0:?}")]
    TypecheckError(metacat::check::Error<OperationKey>),
}

/// Build the graph for one definition without inlining any called definitions.
///
/// Each arrow used by the selected definition remains a single hypergraph box.
/// This is useful for control-flow sketches such as `gpu.tiled-matmul.block-f32`,
/// where primitive GPU arrows should stay opaque instead of being lowered through
/// the existing compiler passes.
pub fn shallow_graph(
    bundle: &TheoryBundle,
    definition: &str,
) -> Result<OpenHypergraph<Obj, Arr>, ShallowError> {
    let key: Operation = definition
        .parse()
        .map_err(|_| ShallowError::InvalidDefinition(definition.to_string()))?;

    let declaration = bundle
        .definitions
        .get(&key)
        .ok_or_else(|| ShallowError::UnknownDefinition(definition.to_string()))?;

    let hexpr = declaration
        .definition
        .clone()
        .ok_or_else(|| ShallowError::UnknownDefinition(definition.to_string()))?;

    let mut term = forget_labels(try_interpret(&bundle.arrow_theory, &hexpr)?);
    let source = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.source_map,
    )?);
    let target = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.target_map,
    )?);

    let checked_nodes = check(&bundle.arrow_theory, source, target, &mut term)
        .map_err(ShallowError::TypecheckError)?;
    let mut term = term.with_nodes(|_| checked_nodes).unwrap();
    term.quotient().map_err(ShallowError::InvalidQuotient)?;
    Ok(term)
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
