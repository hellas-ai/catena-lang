/// Add name.{f} for each arrow f
pub(crate) mod name_symbols;

/// Add const.{type}.{c} arrows for each constant c required.
mod constants;

use hexpr::{Hexpr, interpret::Error as HexprInterpretError};
use metacat::theory::model::SignatureError;
use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError};
use thiserror::Error;

const NAT_THEORY: &str = "nat";
const RESERVED_OPERATION_PREFIXES: &[&str] = &["name.", "const."];
pub(crate) const GENERATED_VARIABLE_PREFIX: &str = "__catena_";
const RESERVED_VARIABLE_PREFIXES: &[&str] = &[GENERATED_VARIABLE_PREFIX];

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
}

pub fn elaborate(mut raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    raw = raw.with_extensions()?;
    check_reserved_operation_prefixes(&raw)?;
    check_reserved_variable_prefixes(&raw)?;
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

fn check_reserved_variable_prefixes(raw: &RawTheorySet) -> Result<(), ElaborateError> {
    for (theory_name, theory) in &raw.theories {
        for (arrow_name, arrow) in &theory.arrows {
            for map in [&arrow.type_maps.0, &arrow.type_maps.1] {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, map)?;
            }
            if let Some(definition) = &arrow.definition {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, definition)?;
            }
        }
    }
    Ok(())
}

fn check_reserved_variables_in_hexpr(
    theory_name: &hexpr::Operation,
    arrow_name: &hexpr::Operation,
    expr: &Hexpr,
) -> Result<(), ElaborateError> {
    match expr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                check_reserved_variables_in_hexpr(theory_name, arrow_name, expr)?;
            }
        }
        Hexpr::Frobenius { sources, targets } => {
            for variable in sources.iter().chain(targets) {
                let variable = variable.to_string();
                if let Some(prefix) = RESERVED_VARIABLE_PREFIXES
                    .iter()
                    .copied()
                    .find(|prefix| variable.starts_with(prefix))
                {
                    return Err(ElaborateError::ReservedVariablePrefix {
                        theory: theory_name.to_string(),
                        arrow: arrow_name.to_string(),
                        variable,
                        prefix,
                    });
                }
            }
        }
        Hexpr::Operation(_) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use metacat::theory::RawTheorySet;

    use super::{ElaborateError, GENERATED_VARIABLE_PREFIX, elaborate};

    #[test]
    fn user_variables_cannot_use_catena_generated_prefix() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr val : 1 -> 1)
              (arr bool : 0 -> 1)
            })

            (theory program type {
              (arr bad : [__catena_p0.] -> (bool val))
            })
            "#,
        )
        .expect("test theory should parse");

        let error = elaborate(raw).expect_err("reserved variable should be rejected");
        assert!(matches!(
            error,
            ElaborateError::ReservedVariablePrefix {
                theory,
                arrow,
                variable,
                prefix,
            } if theory == "program"
                && arrow == "bad"
                && variable == "__catena_p0"
                && prefix == GENERATED_VARIABLE_PREFIX
        ));
    }
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
