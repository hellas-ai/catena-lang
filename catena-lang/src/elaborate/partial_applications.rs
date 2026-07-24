//! Elaborate `partial.f.N` into a CMC closure that captures the first `N`
//! ordinary inputs of `f`.
//!
//! Metavariables and ordinary inputs occupy different boundaries of an
//! interpreted type map:
//!
//! - `source.sources` contains the leading metavariable context from syntax
//!   such as `([n m.] ...)`.
//! - `source.targets` contains the ordinary arrow inputs written inside the
//!   source object.
//!
//! `N` counts only `source.targets`; metavariables are never included in the
//! partial-application count. The generated arrow instead exposes the
//! metavariable context before the captured input prefix:
//!
//! ```text
//! partial.f.N :
//!   metavariables × first-N-inputs
//!     -> (remaining-inputs => outputs)
//! ```
//!
//! For example, if `f` has metavariables `n, t` and ordinary inputs
//! `buffer, index`, then `partial.f.1` is applied to
//! `n-param, t-param, buffer` and returns a closure waiting for `index`.
//! Runtime witnesses appearing among the ordinary inputs still count toward
//! `N`; only the leading type-map context is excluded.
//!
//! The generated definition copies the metavariable context for both the
//! captured side and `name.f`, packs the remaining inputs and outputs as
//! needed, and connects them using `defer`, `tensor`, `compose`, and
//! `name.f lift`.

use hexpr::{Hexpr, Operation, Variable, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};
use open_hypergraphs::lax::OpenHypergraph;

use crate::{
    elaborate::{ElaborateError, packing},
    prefixes::{GENERATED_PARTIAL_PREFIX, GENERATED_VARIABLE_PREFIX, NAME_PREFIX, PARTIAL_PREFIX},
    stdlib::constants::{
        COMPOSE, DEFER, FN_HOM_TYPE, LIFT, PRODUCT_INTRO, PRODUCT_TYPE, TENSOR, UNIT_INTRO,
        UNIT_TYPE,
    },
};

#[derive(Clone)]
struct PartialApplication {
    operation: Operation,
    arrow: Operation,
    applied: usize,
}

type SyntaxTerm = OpenHypergraph<(), Operation>;

pub fn elaborate(raw: &mut RawTheorySet) -> Result<(), ElaborateError> {
    let theory_names = raw.theories.keys().cloned().collect::<Vec<_>>();
    for theory_name in theory_names {
        let Some(theory) = raw.theories.get(&theory_name) else {
            continue;
        };
        let syntax_name = theory.syntax_category.clone();
        if syntax_name.as_str() == super::NAT_THEORY {
            continue;
        }
        let syntax_raw = transitive_dependency_subset([syntax_name.clone()], raw)?;
        let syntax_set = TheorySet::from_raw(syntax_raw)?;
        let syntax = syntax_set
            .theories
            .get(&TheoryId(syntax_name.clone()))
            .ok_or_else(|| {
                ElaborateError::MissingInterpretedSyntaxTheory(syntax_name.to_string())
            })?;

        let theory = raw
            .theories
            .get_mut(&theory_name)
            .ok_or_else(|| ElaborateError::MissingTheory(theory_name.to_string()))?;
        elaborate_theory(theory, syntax)?;
    }
    Ok(())
}

fn elaborate_theory(theory: &mut RawTheory, syntax: &Theory) -> Result<(), ElaborateError> {
    let uses = theory
        .arrows
        .values()
        .filter_map(|arrow| arrow.definition.as_ref())
        .flat_map(partials_in_hexpr)
        .collect::<Vec<_>>();

    for operation in uses {
        let partial = parse_partial(&operation)?;
        if theory.arrows.contains_key(&partial.operation) {
            continue;
        }
        let original = theory.arrows.get(&partial.arrow).cloned().ok_or_else(|| {
            ElaborateError::MissingPartialApplicationArrow {
                theory: theory.name.to_string(),
                operation: partial.operation.to_string(),
                arrow: partial.arrow.to_string(),
            }
        })?;
        let generated = partial_arrows(syntax, &theory.name, &original, &partial)?;
        for arrow in generated {
            theory.arrows.insert(arrow.name.clone(), arrow);
        }
    }
    Ok(())
}

fn partials_in_hexpr(hexpr: &Hexpr) -> Vec<Operation> {
    let mut partials = Vec::new();
    collect_partials(hexpr, &mut partials);
    partials
}

fn collect_partials(hexpr: &Hexpr, partials: &mut Vec<Operation>) {
    match hexpr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                collect_partials(expr, partials);
            }
        }
        Hexpr::Operation(operation) if operation.as_str().starts_with(PARTIAL_PREFIX) => {
            partials.push(operation.clone());
        }
        Hexpr::Frobenius { .. } | Hexpr::Operation(_) => {}
    }
}

fn parse_partial(operation: &Operation) -> Result<PartialApplication, ElaborateError> {
    let suffix = operation
        .as_str()
        .strip_prefix(PARTIAL_PREFIX)
        .unwrap_or_default();
    let Some((arrow, count)) = suffix.rsplit_once('.') else {
        return invalid(operation, "expected `partial.<arrow>.<decimal N>`");
    };
    if arrow.is_empty() || count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
        return invalid(operation, "expected `partial.<arrow>.<decimal N>`");
    }
    let applied = count
        .parse::<usize>()
        .map_err(|_| invalid_error(operation, "argument count does not fit in usize"))?;
    let arrow = arrow
        .parse()
        .map_err(|_| invalid_error(operation, "referenced arrow name is invalid"))?;
    Ok(PartialApplication {
        operation: operation.clone(),
        arrow,
        applied,
    })
}

fn invalid<T>(operation: &Operation, reason: &str) -> Result<T, ElaborateError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(operation: &Operation, reason: &str) -> ElaborateError {
    ElaborateError::InvalidPartialApplication {
        operation: operation.to_string(),
        reason: reason.to_string(),
    }
}

fn partial_arrows(
    syntax: &Theory,
    theory_name: &Operation,
    original: &RawTheoryArrow,
    partial: &PartialApplication,
) -> Result<Vec<RawTheoryArrow>, ElaborateError> {
    let mut source = try_interpret(&syntax.local_signature(), &original.type_maps.0)
        .map_err(|error| ElaborateError::NameSourceTypeMapInterpretation {
            theory: theory_name.to_string(),
            arrow: original.name.to_string(),
            map: original.type_maps.0.clone(),
            error,
        })?
        .map_nodes(|_| ());
    let mut target = try_interpret(&syntax.local_signature(), &original.type_maps.1)
        .map_err(|error| ElaborateError::NameTargetTypeMapInterpretation {
            theory: theory_name.to_string(),
            arrow: original.name.to_string(),
            map: original.type_maps.1.clone(),
            error,
        })?
        .map_nodes(|_| ());
    source.quotient().map_err(|_| {
        invalid_error(
            &partial.operation,
            "source type map has inconsistent named-wire equations",
        )
    })?;
    target.quotient().map_err(|_| {
        invalid_error(
            &partial.operation,
            "target type map has inconsistent named-wire equations",
        )
    })?;
    // The open boundary separates type-map metavariables from arrow inputs.
    // Only the latter participate in the `.N` count.
    let context_arity = source.sources.len();
    let arity = source.targets.len();
    if partial.applied > arity {
        return invalid(
            &partial.operation,
            &format!(
                "cannot apply {} arguments to an arrow with {arity} inputs",
                partial.applied
            ),
        );
    }

    let source_map = partial_source_map(&source, context_arity, partial.applied)?;
    let target_map = partial_target_map(
        &source,
        &target,
        context_arity,
        arity,
        partial.applied,
        target.targets.len(),
    )?;

    let identity_name = generated_helper_name("identity", &partial.operation)?;
    let with_unit_name = generated_helper_name("with-left-unit", &partial.operation)?;
    let definition = partial_definition(
        original,
        context_arity,
        arity,
        partial.applied,
        &identity_name,
        &with_unit_name,
    )?;
    let mut arrows = vec![RawTheoryArrow {
        name: partial.operation.clone(),
        type_maps: (source_map, target_map),
        definition: Some(definition),
    }];

    if partial.applied < arity && partial.applied > 0 {
        let remaining = packed_slice_map(&source, partial.applied, arity)?;
        arrows.push(RawTheoryArrow {
            name: identity_name,
            type_maps: (remaining.clone(), remaining.clone()),
            definition: Some(identity_definition()?),
        });
        arrows.push(RawTheoryArrow {
            name: with_unit_name,
            type_maps: (remaining.clone(), with_left_unit_map(remaining)?),
            definition: Some(with_left_unit_definition()?),
        });
    }
    Ok(arrows)
}

fn partial_source_map(
    source: &SyntaxTerm,
    context: usize,
    applied: usize,
) -> Result<Hexpr, ElaborateError> {
    let mut vars = Vars::default();
    let context_vars = vars.many("context", context)?;
    // A use of `partial.f.N` receives metavariable witnesses first, followed
    // by the captured prefix. Copy the witnesses because the target closure
    // type and the captured call to `f` both depend on them.
    let copied = Hexpr::Frobenius {
        sources: context_vars.clone(),
        targets: context_vars
            .iter()
            .cloned()
            .chain(context_vars.clone())
            .collect(),
    };
    let prefix = slice_map(source, 0, applied, &mut vars)?;
    Ok(Hexpr::Composition(vec![
        copied,
        Hexpr::Tensor(vec![identity(context_vars), prefix]),
    ]))
}

fn partial_target_map(
    source: &SyntaxTerm,
    target: &SyntaxTerm,
    context: usize,
    arity: usize,
    applied: usize,
    target_arity: usize,
) -> Result<Hexpr, ElaborateError> {
    let mut vars = Vars::default();
    let context_vars = vars.many("context", context)?;
    let copied = Hexpr::Frobenius {
        sources: context_vars.clone(),
        targets: context_vars
            .iter()
            .cloned()
            .chain(context_vars.clone())
            .collect(),
    };
    let remaining = pack_after(
        slice_map(source, applied, arity, &mut vars)?,
        arity - applied,
        &mut vars,
    )?;
    let target = pack_after(
        slice_map(target, 0, target_arity, &mut vars)?,
        target_arity,
        &mut vars,
    )?;
    Ok(Hexpr::Composition(vec![
        copied,
        Hexpr::Tensor(vec![remaining, target]),
        op(FN_HOM_TYPE)?,
    ]))
}

fn packed_slice_map(
    source: &SyntaxTerm,
    start: usize,
    end: usize,
) -> Result<Hexpr, ElaborateError> {
    let mut vars = Vars::default();
    let slice = slice_map(source, start, end, &mut vars)?;
    pack_after(slice, end - start, &mut vars)
}

fn slice_map(
    term: &SyntaxTerm,
    start: usize,
    end: usize,
    vars: &mut Vars,
) -> Result<Hexpr, ElaborateError> {
    let node_vars = vars.many("wire", term.hypergraph.nodes.len())?;
    let selected_targets = &term.targets[start..end];
    let mut retained_nodes = vec![false; term.hypergraph.nodes.len()];
    for node in selected_targets {
        retained_nodes[node.0] = true;
    }

    let mut retained_edges = vec![false; term.hypergraph.edges.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (index, boundary) in term.hypergraph.adjacency.iter().enumerate() {
            if retained_edges[index] || !boundary.targets.iter().any(|node| retained_nodes[node.0])
            {
                continue;
            }
            retained_edges[index] = true;
            changed = true;
            for node in &boundary.sources {
                retained_nodes[node.0] = true;
            }
        }
    }

    let mut expressions = vec![Hexpr::Frobenius {
        sources: term
            .sources
            .iter()
            .map(|node| node_vars[node.0].clone())
            .collect(),
        targets: vec![],
    }];
    for (index, operation) in term.hypergraph.edges.iter().enumerate() {
        if !retained_edges[index] {
            continue;
        }
        let boundary = &term.hypergraph.adjacency[index];
        expressions.push(Hexpr::Composition(vec![
            reference(
                &boundary
                    .sources
                    .iter()
                    .map(|node| node_vars[node.0].clone())
                    .collect::<Vec<_>>(),
            ),
            Hexpr::Operation(operation.clone()),
            Hexpr::Frobenius {
                sources: boundary
                    .targets
                    .iter()
                    .map(|node| node_vars[node.0].clone())
                    .collect(),
                targets: vec![],
            },
        ]));
    }
    expressions.push(reference(
        &selected_targets
            .iter()
            .map(|node| node_vars[node.0].clone())
            .collect::<Vec<_>>(),
    ));
    Ok(Hexpr::Composition(expressions))
}

fn pack_after(map: Hexpr, count: usize, vars: &mut Vars) -> Result<Hexpr, ElaborateError> {
    let packed = packing::pack_object(count, &mut || vars.one("pack"))?;
    Ok(Hexpr::Composition(vec![map, packed]))
}

fn with_left_unit_map(remaining: Hexpr) -> Result<Hexpr, ElaborateError> {
    Ok(Hexpr::Composition(vec![
        remaining,
        Hexpr::Tensor(vec![
            op(UNIT_TYPE)?,
            identity(vec![parse_var("unit_payload")?]),
        ]),
        op(PRODUCT_TYPE)?,
    ]))
}

/// Build the CMC term implementing `partial.f.N`.
///
/// Write the packed domain of `f` as `P * R`, where `P` contains the first
/// `N` arguments and `R` contains the remaining arguments. For the nontrivial
/// case `0 < N < arity(f)`, the generated term has the following shape:
///
/// ```text
/// {
///   ({
///     (name.with-left-unit-R lift)
///     ({
///       (P defer)
///       (name.identity-R lift)
///     } tensor)
///   } compose)
///   (name.f lift)
/// }
/// compose
/// ```
///
/// Its types make the construction a little clearer:
///
/// ```text
/// with-left-unit-R              : R       => 1 * R
/// defer(P) tensor identity-R    : 1 * R   => P * R
/// f                             : P * R   => B
/// ------------------------------------------------
/// partial.f.N(P)                : R       => B
/// ```
///
/// This is the general form of the hand-written matrix construction using
/// `index-with-unit`, `input defer`, `index-id`, and `buf-at`.
fn partial_definition(
    original: &RawTheoryArrow,
    context: usize,
    arity: usize,
    applied: usize,
    identity_name: &Operation,
    with_unit_name: &Operation,
) -> Result<Hexpr, ElaborateError> {
    let context_vars = (0..context)
        .map(|index| parse_var(&format!("partial_context_{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let arguments = (0..applied)
        .map(|index| parse_var(&format!("partial_argument_{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = context_vars
        .iter()
        .chain(&arguments)
        .cloned()
        .collect::<Vec<_>>();
    let consume_inputs = Hexpr::Frobenius {
        sources: inputs,
        targets: vec![],
    };
    let original_closure = named_closure(&original.name, &context_vars)?;

    let body = if applied == 0 {
        // Nothing is captured: partial.f.0 is simply the lifted name of f.
        original_closure
    } else {
        // Pack the supplied prefix P into one object, then capture it as a
        // closure 1 => P.
        let captured = Hexpr::Composition(vec![pack_values(&arguments)?, op(DEFER)?]);
        if applied == arity {
            // There is no remaining R. Compose 1 => P with P => B directly,
            // producing the fully captured closure 1 => B.
            Hexpr::Composition(vec![
                Hexpr::Tensor(vec![captured, original_closure]),
                op(COMPOSE)?,
            ])
        } else {
            // These are the generalized `index-id` and `index-with-unit`
            // arrows from the motivating matrix example.
            let identity_closure = named_closure(identity_name, &context_vars)?;
            let with_unit_closure = named_closure(with_unit_name, &context_vars)?;

            // (1 => P) tensor (R => R) gives (1 * R) => (P * R).
            let append_capture = Hexpr::Composition(vec![
                Hexpr::Tensor(vec![captured, identity_closure]),
                op(TENSOR)?,
            ]);

            // Precompose with R => (1 * R), obtaining R => (P * R).
            let prepare = Hexpr::Composition(vec![
                Hexpr::Tensor(vec![with_unit_closure, append_capture]),
                op(COMPOSE)?,
            ]);

            // Finally compose R => (P * R) with the lifted original function
            // (P * R) => B.
            Hexpr::Composition(vec![
                Hexpr::Tensor(vec![prepare, original_closure]),
                op(COMPOSE)?,
            ])
        }
    };
    Ok(Hexpr::Composition(vec![consume_inputs, body]))
}

fn named_closure(name: &Operation, context_vars: &[Variable]) -> Result<Hexpr, ElaborateError> {
    Ok(Hexpr::Composition(vec![
        reference(context_vars),
        op(&format!("{NAME_PREFIX}{name}"))?,
        op(LIFT)?,
    ]))
}

fn pack_values(values: &[Variable]) -> Result<Hexpr, ElaborateError> {
    match values {
        [] => op(UNIT_INTRO),
        [only] => Ok(reference(std::slice::from_ref(only))),
        [head @ .., last] => Ok(Hexpr::Composition(vec![
            Hexpr::Tensor(vec![
                pack_values(head)?,
                reference(std::slice::from_ref(last)),
            ]),
            op(PRODUCT_INTRO)?,
        ])),
    }
}

fn identity_definition() -> Result<Hexpr, ElaborateError> {
    let wire = parse_var("partial_identity")?;
    Ok(Hexpr::Frobenius {
        sources: vec![wire.clone()],
        targets: vec![wire],
    })
}

fn with_left_unit_definition() -> Result<Hexpr, ElaborateError> {
    let wire = parse_var("partial_with_unit")?;
    Ok(Hexpr::Composition(vec![
        Hexpr::Frobenius {
            sources: vec![wire.clone()],
            targets: vec![],
        },
        Hexpr::Tensor(vec![op(UNIT_INTRO)?, reference(&[wire])]),
        op(PRODUCT_INTRO)?,
    ]))
}

fn generated_helper_name(kind: &str, partial: &Operation) -> Result<Operation, ElaborateError> {
    format!("{GENERATED_PARTIAL_PREFIX}{kind}.{}", partial.as_str())
        .parse()
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(partial.to_string()))
}

fn reference(vars: &[Variable]) -> Hexpr {
    Hexpr::Frobenius {
        sources: vec![],
        targets: vars.to_vec(),
    }
}

fn identity(vars: Vec<Variable>) -> Hexpr {
    Hexpr::Frobenius {
        sources: vars.clone(),
        targets: vars,
    }
}

fn op(name: &str) -> Result<Hexpr, ElaborateError> {
    name.parse()
        .map(Hexpr::Operation)
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(name.to_string()))
}

fn parse_var(name: &str) -> Result<Variable, ElaborateError> {
    name.parse()
        .map_err(|_| ElaborateError::InvalidGeneratedVariable(name.to_string()))
}

#[derive(Default)]
struct Vars(usize);

impl Vars {
    fn one(&mut self, stem: &str) -> Result<Variable, ElaborateError> {
        let variable = parse_var(&format!(
            "{GENERATED_VARIABLE_PREFIX}partial_{stem}_{}",
            self.0
        ))?;
        self.0 += 1;
        Ok(variable)
    }

    fn many(&mut self, stem: &str, count: usize) -> Result<Vec<Variable>, ElaborateError> {
        (0..count).map(|_| self.one(stem)).collect()
    }
}

#[cfg(test)]
mod tests {
    use metacat::theory::{RawTheorySet, TheorySet};

    use crate::elaborate::{ElaborateError, elaborate};

    fn fixture(use_operation: &str) -> RawTheorySet {
        RawTheorySet::from_text(&format!(
            r#"
            (theory type nat {{
              (arr 1 : 0 -> 1)
              (arr * : 2 -> 1)
              (arr -> : 2 -> 1)
              (arr => : 2 -> 1)
              (arr val : 1 -> 1)
              (arr bool : 0 -> 1)
            }})
            (theory program type {{
              (arr unit.intro : [] -> 1)
              (arr *.intro : [a b] -> *)
              (arr defer : [a] -> ({{1 [a]}} =>))
              (arr compose : {{[a b c.]
                ([. a b] =>)
                ([. b c] =>)
              }} -> ([a b c . a c] =>))
              (arr tensor : {{[a0 b0 a1 b1.]
                ([. a0 b0] =>)
                ([. a1 b1] =>)
              }} -> {{[a0 b0 a1 b1.]
                ([. a0 a1] * [src.])
                ([. b0 b1] * [tgt.])
                ([. src tgt] =>)
              }})
              (arr lift : (-> val) -> =>)
              (arr f : {{(bool val) (bool val) (bool val)}} -> (bool val))
              (def use : {{(bool val) (bool val)}} -> ({{(bool val) (bool val)}} =>) = {use_operation})
            }})
            "#
        ))
        .expect("fixture should parse")
    }

    fn dependent_fixture(arrows: &str) -> RawTheorySet {
        RawTheorySet::from_text(&format!(
            r#"
            (theory type nat {{
              (arr 1 : 0 -> 1)
              (arr * : 2 -> 1)
              (arr -> : 2 -> 1)
              (arr => : 2 -> 1)
              (arr val : 1 -> 1)
              (arr : : 2 -> 1)
              (arr bool : 0 -> 1)
              (arr u64 : 0 -> 1)
              (arr box : 1 -> 1)
            }})
            (theory program type {{
              (arr unit.intro : [] -> 1)
              (arr *.intro : [a b] -> *)
              (arr defer : [a] -> ({{1 [a]}} =>))
              (arr compose : {{[a b c.]
                ([. a b] =>)
                ([. b c] =>)
              }} -> ([a b c . a c] =>))
              (arr tensor : {{[a0 b0 a1 b1.]
                ([. a0 b0] =>)
                ([. a1 b1] =>)
              }} -> {{[a0 b0 a1 b1.]
                ([. a0 a1] * [src.])
                ([. b0 b1] * [tgt.])
                ([. src tgt] =>)
              }})
              (arr lift : (-> val) -> =>)
              {arrows}
            }})
            "#
        ))
        .expect("dependent fixture should parse")
    }

    #[test]
    fn generates_and_typechecks_a_used_partial_application() {
        let elaborated = elaborate(fixture("partial.f.2")).expect("partial should elaborate");
        let program: hexpr::Operation = "program".parse().unwrap();
        let partial: hexpr::Operation = "partial.f.2".parse().unwrap();
        let arrow = &elaborated.theories[&program].arrows[&partial];
        assert!(arrow.definition.is_some());
        let interpreted = TheorySet::from_raw(elaborated).expect("generated theory should load");
        crate::check::check(&interpreted).expect("generated CMC definition should typecheck");
    }

    #[test]
    fn does_not_generate_unused_partial_applications() {
        let elaborated = elaborate(fixture("{[a] [.b]}")).expect("fixture should elaborate");
        let program: hexpr::Operation = "program".parse().unwrap();
        assert!(
            !elaborated.theories[&program]
                .arrows
                .keys()
                .any(|name| name.as_str().starts_with("partial."))
        );
    }

    #[test]
    fn rejects_applying_too_many_arguments() {
        let error = elaborate(fixture("partial.f.4")).expect_err("invalid partial should fail");
        assert!(matches!(
            error,
            ElaborateError::InvalidPartialApplication { .. }
        ));
    }

    #[test]
    fn supports_zero_and_full_partial_application() {
        for operation in ["partial.f.0", "partial.f.3"] {
            let elaborated = elaborate(fixture(operation)).expect("partial should elaborate");
            let program: hexpr::Operation = "program".parse().unwrap();
            let operation: hexpr::Operation = operation.parse().unwrap();
            assert!(
                elaborated.theories[&program]
                    .arrows
                    .contains_key(&operation)
            );
        }
    }

    #[test]
    fn metavariables_precede_captured_inputs_and_do_not_count_toward_n() {
        let raw = dependent_fixture(
            r#"
            (arr f :
              ([n m.] {
                ([.n] box)
                (bool val)
              })
              ->
              ([n m.] ([.m] box)))
            (def use :
              ([n m.] {
                [.n]
                [.m]
                ([.n] box)
              })
              ->
              ([n m.] ({(bool val) ([.m] box)} =>))
              = partial.f.1)
            "#,
        );

        let elaborated = elaborate(raw).expect("dependent partial should elaborate");
        let interpreted = TheorySet::from_raw(elaborated).expect("generated theory should load");
        crate::check::check(&interpreted)
            .expect("the generated arrow should accept n and m before the captured first input");
    }

    #[test]
    fn applying_too_many_inputs_ignores_metavariables() {
        let raw = dependent_fixture(
            r#"
            (arr f :
              ([n m.] {
                ([.n] box)
                (bool val)
              })
              ->
              ([n m.] ([.m] box)))
            (def use : [] -> [] = partial.f.3)
            "#,
        );

        let error = elaborate(raw).expect_err("two inputs plus two metavars still has arity two");
        let ElaborateError::InvalidPartialApplication { reason, .. } = error else {
            panic!("expected an invalid partial application");
        };
        assert!(
            reason.contains("arrow with 2 inputs"),
            "unexpected error: {reason}"
        );
    }

    #[test]
    fn runtime_witnesses_in_the_source_are_counted_toward_n() {
        let raw = dependent_fixture(
            r#"
            (arr f :
              ([n.] {
                ({[.n] u64} :)
                (bool val)
              })
              ->
              ([n.] (bool val)))
            (def use :
              ([n.] {
                [.n]
                ({[.n] u64} :)
              })
              ->
              ([n.] ({(bool val) (bool val)} =>))
              = partial.f.1)
            "#,
        );

        let elaborated = elaborate(raw).expect("runtime witness should be the captured input");
        let interpreted = TheorySet::from_raw(elaborated).expect("generated theory should load");
        crate::check::check(&interpreted)
            .expect("partial.f.1 should leave only the bool input for the closure");
    }

    #[test]
    fn rejects_malformed_and_missing_partial_targets() {
        for operation in ["partial.f.nope", "partial.missing.1"] {
            let error = elaborate(fixture(operation)).expect_err("invalid partial should fail");
            assert!(matches!(
                error,
                ElaborateError::InvalidPartialApplication { .. }
                    | ElaborateError::MissingPartialApplicationArrow { .. }
            ));
        }
    }

    #[test]
    fn preserves_dependent_source_context() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory type nat {
              (arr 1 : 0 -> 1)
              (arr * : 2 -> 1)
              (arr -> : 2 -> 1)
              (arr => : 2 -> 1)
              (arr val : 1 -> 1)
              (arr box : 1 -> 1)
            })
            (theory program type {
              (arr unit.intro : [] -> 1)
              (arr *.intro : [a b] -> *)
              (arr defer : [a] -> ({1 [a]} =>))
              (arr compose : {[a b c.]
                ([. a b] =>)
                ([. b c] =>)
              } -> ([a b c . a c] =>))
              (arr tensor : {[a0 b0 a1 b1.]
                ([. a0 b0] =>)
                ([. a1 b1] =>)
              } -> {[a0 b0 a1 b1.]
                ([. a0 a1] * [src.])
                ([. b0 b1] * [tgt.])
                ([. src tgt] =>)
              })
              (arr lift : (-> val) -> =>)
              (arr f : ([a.] {([.a] box) ([.a] box)}) -> ([a.] ([.a] box)))
              (def use :
                ([a.] {[.a] ([.a] box)})
                ->
                ([a.] {([.a] box) ([.a] box)} =>)
                = partial.f.1)
            })
            "#,
        )
        .expect("fixture should parse");

        let elaborated = elaborate(raw).expect("dependent partial should elaborate");
        let interpreted = TheorySet::from_raw(elaborated).expect("generated theory should load");
        crate::check::check(&interpreted).expect("dependent partial should typecheck");
    }
}
