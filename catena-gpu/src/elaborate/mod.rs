//! Minimal source elaboration: extend theories and generate `name.*` arrows.

mod names;

use hexpr::{Hexpr, Operation, interpret::Error as HexprInterpretError};
use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError, model::SignatureError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ElaborateError {
    #[error(transparent)]
    Extensions(#[from] ExtensionsError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
    #[error("missing theory `{0}` during elaboration")]
    MissingTheory(String),
    #[error("missing interpreted syntax theory `{0}` during elaboration")]
    MissingSyntax(String),
    #[error("invalid generated operation `{0}`")]
    InvalidOperation(String),
    #[error("invalid generated variable `{0}`")]
    InvalidVariable(String),
    #[error("failed to interpret source type map for `{theory}.{arrow}` from `{map}`: {error}")]
    SourceTypeMap {
        theory: String,
        arrow: String,
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
    #[error("failed to interpret target type map for `{theory}.{arrow}` from `{map}`: {error}")]
    TargetTypeMap {
        theory: String,
        arrow: String,
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
}

/// Elaborate only generated function-name arrows.
pub fn elaborate(raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    let mut raw = raw.with_extensions()?;
    let theories = raw
        .theories
        .iter()
        .filter(|(_, theory)| theory.syntax_category.as_str() != "nat")
        .map(|(name, _)| name.clone())
        .collect::<Vec<Operation>>();
    for theory in theories {
        names::elaborate_theory(&mut raw, &theory)?;
    }
    Ok(raw)
}
