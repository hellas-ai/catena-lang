mod name_symbols;

use hexpr::{Hexpr, interpret::Error as HexprInterpretError};
use metacat::theory::model::SignatureError;
use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError};
use thiserror::Error;

const NAT_THEORY: &str = "nat";

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
    MissingInterpretedSyntaxTheory(String),
    #[error("generated operation name `{0}` is invalid")]
    InvalidGeneratedOperation(String),
    #[error("generated variable name `{0}` is invalid")]
    InvalidGeneratedVariable(String),
    #[error(
        "failed to interpret source type map for `name.{theory}.{arrow}` from `{map}`: {error}"
    )]
    NameSourceTypeMapInterpretation {
        theory: String,
        arrow: String,
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
    #[error(
        "failed to interpret target type map for `name.{theory}.{arrow}` from `{map}`: {error}"
    )]
    NameTargetTypeMapInterpretation {
        theory: String,
        arrow: String,
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
}

pub fn elaborate(mut raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    raw = raw.with_extensions()?;

    let theory_names: Vec<_> = raw
        .theories
        .iter()
        .filter(|(_, theory)| theory.syntax_category.as_str() != NAT_THEORY)
        .map(|(name, _)| name.clone())
        .collect();

    for theory_name in theory_names {
        name_symbols::elaborate_theory(&mut raw, &theory_name)?;
    }

    Ok(raw)
}
