//! Elaboration passes for catena.
//!
//! An unordered list of passes:
//!
//! - Mutually interleave arrows from two theories
//! - (TODO): For each `f : A -> B` in a theory, add a 'name': `name.f : I -> (A => B)`
pub(crate) mod interleave_arrows;

use metacat::theory::{RawTheorySet, Theory, TheoryId, TheorySet};

use crate::check::CheckError;
use interleave_arrows::interleave;

const SYNTAX_THEORY: &str = "syntax";
const NAT_THEORY: &str = "nat";

/// Elaborate input program to interleave control/data maps.
pub fn elaborate(raw: RawTheorySet) -> Result<RawTheorySet, CheckError> {
    let syntax = interpret_syntax(&raw)?;
    Ok(interleave(&syntax, raw)?)
}

fn interpret_syntax(raw: &RawTheorySet) -> Result<Theory, CheckError> {
    let syntax_name: hexpr::Operation = SYNTAX_THEORY.parse().expect("valid syntax theory name");
    let syntax_raw = raw
        .theories
        .get(&syntax_name)
        .ok_or_else(|| CheckError::MissingSyntaxTheory(SYNTAX_THEORY.to_string()))?;

    let mut subset = RawTheorySet {
        theories: Default::default(),
        extensions: Vec::new(),
    };

    let mut current = Some(syntax_raw);
    while let Some(theory) = current {
        if subset.theories.contains_key(&theory.name) {
            break;
        }
        subset.theories.insert(theory.name.clone(), theory.clone());
        current = if theory.syntax_category.as_str() == NAT_THEORY {
            None
        } else {
            raw.theories.get(&theory.syntax_category)
        };
    }

    let interpreted = TheorySet::from_raw(subset)?;
    interpreted
        .theories
        .get(&TheoryId(syntax_name))
        .cloned()
        .ok_or_else(|| CheckError::MissingInterpretedSyntaxTheory(SYNTAX_THEORY.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        check::{CheckError, check},
        elaborate::interleave_arrows::InterleaveError,
    };
    use metacat::theory::model::SignatureError;

    #[test]
    fn elaborate_then_check_interleaves_then_typechecks() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory syntax nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr + : 2 -> 1)
              (arr 0 : 0 -> 1)
              (arr f32 : 0 -> 1)
            })

            (theory data syntax {
              (arr f32.add : {f32 f32} -> f32)

              # after interleaving, this should typecheck
              (def merge : ({1 1} +) -> 1 = control.merge)

            })

            (theory control syntax {
                (arr merge : ({1 1} +) -> 1)

                # after interleaving, this should typecheck
                (def expected : ({f32 f32} *) -> f32 = data.f32.add)
            })
            "#,
        )
        .unwrap();

        let elaborated_raw = elaborate(raw).unwrap();
        let elaborated = check(&elaborated_raw).unwrap();
        assert!(
            elaborated
                .theories
                .get(&TheoryId("control".parse().unwrap()))
                .and_then(|theory| theory.get_arrow(&"data.f32.add".parse().unwrap()))
                .is_some()
        );
    }

    #[test]
    fn elaborate_surfaces_interleave_type_map_errors() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory syntax nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr + : 2 -> 1)
              (arr 0 : 0 -> 1)
              (arr f32 : 0 -> 1)
            })

            (theory data syntax {
              # bad: type is invalid - no such constructor 'value'
              (arr bad : value -> f32)
            })

            (theory control syntax {
              (arr pass : 1 -> 1)
            })
            "#,
        )
        .unwrap();

        let error = elaborate(raw).expect_err("elaboration should return an error");
        match error {
            CheckError::Interleave(InterleaveError::BoundaryTypeMapInterpretation {
                map,
                error:
                    hexpr::interpret::Error::Signature(op, SignatureError::NoSuchOperation(missing)),
            }) => {
                assert_eq!(op.as_str(), "value");
                assert_eq!(missing.as_str(), "value");
                assert_eq!(map.to_string(), "value");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn elaborate_errors_when_control_theory_is_missing() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory syntax nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr + : 2 -> 1)
              (arr 0 : 0 -> 1)
              (arr f32 : 0 -> 1)
            })

            (theory data syntax {
              (arr f32.add : {f32 f32} -> f32)
            })
            "#,
        )
        .unwrap();

        let error = elaborate(raw).expect_err("elaboration should fail without control");
        match error {
            CheckError::Interleave(InterleaveError::MissingTheory(theory)) => {
                assert_eq!(theory.as_str(), "control");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn elaborate_errors_when_data_theory_is_missing() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory syntax nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr + : 2 -> 1)
              (arr 0 : 0 -> 1)
              (arr f32 : 0 -> 1)
            })

            (theory control syntax {
              (arr branch : 1 -> f32)
            })
            "#,
        )
        .unwrap();

        let error = elaborate(raw).expect_err("elaboration should fail without data");
        match error {
            CheckError::Interleave(InterleaveError::MissingTheory(theory)) => {
                assert_eq!(theory.as_str(), "data");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn elaborate_errors_when_lifted_arrow_already_exists() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory syntax nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr + : 2 -> 1)
              (arr 0 : 0 -> 1)
              (arr f32 : 0 -> 1)
            })

            (theory data syntax {
              (arr f32.add : {f32 f32} -> f32)
            })

            (theory control syntax {
              (arr data.f32.add : f32 -> f32)
            })
            "#,
        )
        .unwrap();

        let error = elaborate(raw).expect_err("elaboration should fail on duplicate lifted arrow");
        match error {
            CheckError::Interleave(InterleaveError::DuplicateLiftedArrow { theory, arrow }) => {
                assert_eq!(theory.as_str(), "control");
                assert_eq!(arrow.as_str(), "data.f32.add");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
