/// Add name.{f} for each arrow f
mod name_symbols;

/// Add const.{type}.{c} arrows for each constant c required.
mod constants;

use hexpr::{Hexpr, interpret::Error as HexprInterpretError};
use metacat::theory::model::SignatureError;
use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError};
use thiserror::Error;

const NAT_THEORY: &str = "nat";
const RESERVED_OPERATION_PREFIXES: &[&str] = &["name.", "const."];

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
}

pub fn elaborate(mut raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    raw = raw.with_extensions()?;
    check_reserved_operation_prefixes(&raw)?;
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

fn check_reserved_operation_prefixes(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    for (theory_name, theory) in &raw.theories {
        for arrow_name in theory.arrows.keys() {
            if let Some(prefix) = RESERVED_OPERATION_PREFIXES
                .iter()
                .copied()
                .find(|prefix| arrow_name.as_str().starts_with(prefix))
            {
                return Err(ElaborateError::ReservedOperationPrefix {
                    theory: theory_name.to_string(),
                    arrow: arrow_name.to_string(),
                    prefix,
                });
            }
        }
    }
    Ok(())
}
