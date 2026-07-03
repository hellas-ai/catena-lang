/// Add name.{f} for each arrow f
pub(crate) mod name_symbols;

/// Add const.{type}.{c} arrows for each constant c required.
mod constants;

mod validate;

use hexpr::{Hexpr, interpret::Error as HexprInterpretError};
use metacat::theory::model::SignatureError;
use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError};
use thiserror::Error;

pub(crate) const NAT_THEORY: &str = "nat";

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
    #[error("operation `{theory}.{arrow}` uses reserved prefix `{prefix}`")]
    ReservedOperationPrefix {
        theory: String,
        arrow: String,
        prefix: &'static str,
    },
    #[error("variable `{theory}.{arrow}:{variable}` uses reserved prefix `{prefix}`")]
    ReservedVariablePrefix {
        theory: String,
        arrow: String,
        variable: String,
        prefix: &'static str,
    },
    #[error("invalid integer constant `{operation}`: {reason}")]
    InvalidConstant { operation: String, reason: String },
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
    #[error(
        "arrow `{theory}.{arrow}` source and target type maps must have the same context domain: source has `{source_domain}`, target has `{target_domain}`"
    )]
    TypeMapDomainMismatch {
        theory: String,
        arrow: String,
        source_domain: String,
        target_domain: String,
    },
}

pub fn elaborate(mut raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    raw = raw.with_extensions()?;
    validate::pre_elaboration_invariants(&raw)?;
    constants::elaborate(&mut raw, constants::U64)?;
    constants::elaborate(&mut raw, constants::U32)?;

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
