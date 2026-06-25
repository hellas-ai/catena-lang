//! Elaborate a theory by adding a symbol `name.f : I -> (A -> B)` for each arrow `f : A -> B`.
//! This follows from "finitary closed monoidal categories".
use hexpr::{Hexpr, Operation, Variable, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};

use crate::elaborate::{ElaborateError, GENERATED_VARIABLE_PREFIX};

const FN_TYPE: &str = "->";
const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const VALUE_TYPE: &str = "val";
const NAME_PREFIX: &str = "name.";

#[derive(Default)]
struct GeneratedVars {
    next: usize,
}

impl GeneratedVars {
    fn var(&mut self, stem: &str) -> Result<Variable, ElaborateError> {
        let name = format!("{GENERATED_VARIABLE_PREFIX}{stem}{}", self.next);
        self.next += 1;
        parse_variable(&name)
    }

    fn vars(&mut self, stem: &str, arity: usize) -> Result<Vec<Variable>, ElaborateError> {
        (0..arity).map(|_| self.var(stem)).collect()
    }
}

pub fn elaborate_theory(
    raw: &mut RawTheorySet,
    theory_name: &Operation,
) -> Result<(), ElaborateError> {
    let theory = raw
        .theories
        .get(theory_name)
        .ok_or_else(|| ElaborateError::MissingTheory(theory_name.to_string()))?;

    let syntax_theory_name = theory.syntax_category.clone();
    let raw_syntax_dependencies = transitive_dependency_subset([syntax_theory_name.clone()], raw)?;
    let syntax_dependencies = TheorySet::from_raw(raw_syntax_dependencies)?;
    let syntax = syntax_dependencies
        .theories
        .get(&TheoryId(syntax_theory_name))
        .ok_or_else(|| {
            ElaborateError::MissingInterpretedSyntaxTheory(theory.syntax_category.to_string())
        })?;

    let theory = raw
        .theories
        .get_mut(theory_name)
        .ok_or_else(|| ElaborateError::MissingTheory(theory_name.to_string()))?;
    elaborate_theory_with_interpreted_syntax(theory, syntax)?;
    Ok(())
}

fn elaborate_theory_with_interpreted_syntax(
    raw: &mut RawTheory,
    syntax: &Theory,
) -> Result<(), ElaborateError> {
    let mut new_arrows = Vec::new();
    for arrow in raw.arrows.values() {
        new_arrows.push(name_arrow(syntax, &raw.name, arrow)?);
    }

    for arrow in new_arrows {
        raw.arrows.insert(arrow.name.clone(), arrow);
    }
    Ok(())
}

pub(crate) fn name_arrow(
    syntax: &Theory,
    theory_name: &Operation,
    raw: &RawTheoryArrow,
) -> Result<RawTheoryArrow, ElaborateError> {
    Ok(RawTheoryArrow {
        name: parse_operation(&format!("{NAME_PREFIX}{}", raw.name))?,
        type_maps: (
            source_type_map(syntax, theory_name, raw)?,
            target_type_map(syntax, theory_name, raw)?,
        ),
        definition: None,
    })
}

fn source_type_map(
    syntax: &Theory,
    theory_name: &Operation,
    raw: &RawTheoryArrow,
) -> Result<Hexpr, ElaborateError> {
    let interpreted_source =
        try_interpret(&syntax.local_signature(), &raw.type_maps.0).map_err(|error| {
            ElaborateError::NameSourceTypeMapInterpretation {
                theory: theory_name.to_string(),
                arrow: raw.name.to_string(),
                map: raw.type_maps.0.clone(),
                error,
            }
        })?;
    let mut generated = GeneratedVars::default();
    let metavars = generated.vars("p", interpreted_source.sources.len())?;

    Ok(Hexpr::Frobenius {
        sources: metavars.clone(),
        targets: metavars,
    })
}

fn target_type_map(
    syntax: &Theory,
    theory_name: &Operation,
    raw: &RawTheoryArrow,
) -> Result<Hexpr, ElaborateError> {
    let interpreted_source =
        try_interpret(&syntax.local_signature(), &raw.type_maps.0).map_err(|error| {
            ElaborateError::NameSourceTypeMapInterpretation {
                theory: theory_name.to_string(),
                arrow: raw.name.to_string(),
                map: raw.type_maps.0.clone(),
                error,
            }
        })?;
    let interpreted_target =
        try_interpret(&syntax.local_signature(), &raw.type_maps.1).map_err(|error| {
            ElaborateError::NameTargetTypeMapInterpretation {
                theory: theory_name.to_string(),
                arrow: raw.name.to_string(),
                map: raw.type_maps.1.clone(),
                error,
            }
        })?;

    let mut generated = GeneratedVars::default();
    let metavars = generated.vars("p", interpreted_source.sources.len())?;
    let mut copied_metavars = metavars.clone();
    copied_metavars.extend(metavars.clone());
    let copy = Hexpr::Frobenius {
        sources: metavars,
        targets: copied_metavars,
    };

    let pack_s = Hexpr::Composition(vec![
        raw.type_maps.0.clone(),
        pack_object(&mut generated, "s", interpreted_source.targets.len())?,
    ]);
    let pack_t = Hexpr::Composition(vec![
        raw.type_maps.1.clone(),
        pack_object(&mut generated, "t", interpreted_target.targets.len())?,
    ]);

    Ok(Hexpr::Composition(vec![
        copy,
        Hexpr::Tensor(vec![pack_s, pack_t]),
        parse_operation_hexpr(FN_TYPE)?,
        parse_operation_hexpr(VALUE_TYPE)?,
    ]))
}

fn pack_object(
    generated: &mut GeneratedVars,
    stem: &str,
    object_size: usize,
) -> Result<Hexpr, ElaborateError> {
    match object_size {
        0 => parse_operation_hexpr(UNIT_TYPE),
        1 => Ok(identity_var(generated.var(stem)?)),
        2 => parse_operation_hexpr(PRODUCT_TYPE),
        n => Ok(Hexpr::Composition(vec![
            Hexpr::Tensor(vec![
                pack_object(generated, stem, n - 1)?,
                identity_var(generated.var(stem)?),
            ]),
            parse_operation_hexpr(PRODUCT_TYPE)?,
        ])),
    }
}

fn parse_variable(name: &str) -> Result<Variable, ElaborateError> {
    name.parse()
        .map_err(|_| ElaborateError::InvalidGeneratedVariable(name.to_string()))
}

fn identity_var(var: Variable) -> Hexpr {
    Hexpr::Frobenius {
        sources: vec![var.clone()],
        targets: vec![var],
    }
}

fn parse_operation(name: &str) -> Result<Operation, ElaborateError> {
    name.parse::<Operation>()
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(name.to_string()))
}

fn parse_operation_hexpr(name: &str) -> Result<Hexpr, ElaborateError> {
    Ok(Hexpr::Operation(parse_operation(name)?))
}

#[cfg(test)]
mod tests {
    use metacat::theory::RawTheorySet;

    use crate::elaborate::elaborate;

    fn assert_generated_arrow_type_maps(raw_text: &str, expected_text: &str, arrow_name: &str) {
        let raw = RawTheorySet::from_text(raw_text).expect("test theory should parse");
        let elaborated = elaborate(raw).expect("test theory should elaborate");
        let expected =
            RawTheorySet::from_text(expected_text).expect("expected theory should parse");

        let program: super::Operation = "program".parse().unwrap();
        let arrow_name: super::Operation = arrow_name.parse().unwrap();
        let actual_arrow = elaborated
            .theories
            .get(&program)
            .and_then(|theory| theory.arrows.get(&arrow_name))
            .expect("generated arrow should exist");
        let expected_arrow = expected
            .theories
            .get(&program)
            .and_then(|theory| theory.arrows.get(&arrow_name))
            .expect("expected arrow should exist");

        assert_eq!(actual_arrow.type_maps, expected_arrow.type_maps);
    }

    #[test]
    fn name_target_type_map_uses_globally_fresh_generated_vars() {
        assert_generated_arrow_type_maps(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr * : 2 -> 1)
              (arr -> : 2 -> 1)
              (arr val : 1 -> 1)
              (arr bool : 0 -> 1)
              (arr ix : 1 -> 1)
            })

            (theory program type {
              (arr dep : {[n.] ([.n] ix val)} -> {[n.] (bool val)})
            })
            "#,
            r#"
            (theory program type {
              (arr name.dep :
                [__catena_p0 . __catena_p0]
                ->
                ([__catena_p0 . __catena_p0 __catena_p0]
                  {
                    ({[n.] ([.n] ix val)} [__catena_s1 . __catena_s1])
                    ({[n.] (bool val)} [__catena_t2 . __catena_t2])
                  }
                  ->
                  val))
            })
            "#,
            "name.dep",
        );
    }

    #[test]
    fn name_payload_target_type_map_uses_globally_fresh_generated_vars() {
        assert_generated_arrow_type_maps(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr * : 2 -> 1)
              (arr -> : 2 -> 1)
              (arr val : 1 -> 1)
              (arr bool : 0 -> 1)
              (arr ix : 1 -> 1)
            })

            (theory program type {
              (arr dep.payload.then :
                {[n.]
                  (bool val)
                  ([.n] ix val)
                }
                ->
                {[n.]
                  (bool val)
                })
            })
            "#,
            r#"
            (theory program type {
              (arr name.dep.payload.then :
                [__catena_p0 . __catena_p0]
                ->
                ([__catena_p0 . __catena_p0 __catena_p0]
                  {
                    ({[n.] (bool val) ([.n] ix val)} *)
                    ({[n.] (bool val)} [__catena_t1 . __catena_t1])
                  }
                  ->
                  val))
            })
            "#,
            "name.dep.payload.then",
        );
    }
}
