use hexpr::Operation;
use metacat::{check::check, theory::Theory};
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
    #[error("Typecheck failed: {0:?}")]
    TypecheckError(metacat::check::Error<Operation>),
}

pub fn shallow_graph(
    theory: &Theory,
    definition: &str,
) -> Result<OpenHypergraph<Obj, Arr>, ShallowError> {
    let key: Operation = definition
        .parse()
        .map_err(|_| ShallowError::InvalidDefinition(definition.to_string()))?;

    let arrow = theory
        .get_arrow(&key)
        .ok_or_else(|| ShallowError::UnknownDefinition(definition.to_string()))?;
    let mut term = arrow
        .definition
        .clone()
        .ok_or_else(|| ShallowError::UnknownDefinition(definition.to_string()))?;

    let checked_nodes = check(
        theory,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        &mut term,
    )
    .map_err(ShallowError::TypecheckError)?;
    let mut term = term.with_nodes(|_| checked_nodes).unwrap();
    term.quotient().map_err(ShallowError::InvalidQuotient)?;
    Ok(term)
}
