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

use hexpr::{Hexpr, Operation, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};
use open_hypergraphs::lax::{EdgeId, NodeId, OpenHypergraph};

use crate::{
    elaborate::ElaborateError,
    hexpr::term_to_hexpr,
    nonstrict::pack_nodes,
    prefixes::{GENERATED_PARTIAL_PREFIX, NAME_PREFIX, PARTIAL_PREFIX},
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
        Hexpr::Frobenius { .. } | Hexpr::Hole | Hexpr::Wire(_) | Hexpr::Operation(_) => {}
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

/// Generate the arrows needed to implement one use of `partial.f.N`.
///
/// The main generated arrow captures the first `N` inputs of `f`:
///
/// ```text
/// partial.f.N : A_0, ..., A_(N-1) -> (R => B)
/// ```
///
/// where `R = pack(A_N, ..., A_(k-1))` is the packed remainder of `f`'s
/// domain.
///
/// When the captured prefix and `R` are both nonempty, let
/// `D_i = pack(A_(i+1), ..., A_(k-1))`. For each captured input `A_i`, the CMC
/// construction also needs two ordinary functions which can be named and
/// lifted into closures:
///
/// ```text
/// catena.partial.identity.i.partial.f.N       : D_i -> D_i
/// catena.partial.with-left-unit.i.partial.f.N : D_i -> 1 * D_i
/// ```
///
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

    let helper_names = if partial.applied < arity {
        (0..partial.applied)
            .map(|index| {
                Ok((
                    generated_helper_name(&format!("identity.{index}"), &partial.operation)?,
                    generated_helper_name(&format!("with-left-unit.{index}"), &partial.operation)?,
                ))
            })
            .collect::<Result<Vec<_>, ElaborateError>>()?
    } else {
        Vec::new()
    };
    let definition = partial_definition(
        original,
        context_arity,
        arity,
        partial.applied,
        &helper_names,
    )?;
    let mut arrows = vec![RawTheoryArrow {
        name: partial.operation.clone(),
        type_maps: (source_map, target_map),
        definition: Some(definition),
    }];

    if partial.applied < arity && partial.applied > 0 {
        for (index, (identity_name, with_unit_name)) in helper_names.into_iter().enumerate() {
            let suffix = packed_slice_term(&source, index + 1, arity);
            let suffix_map = term_to_hexpr(&suffix);
            arrows.push(RawTheoryArrow {
                name: identity_name,
                type_maps: (suffix_map.clone(), suffix_map),
                definition: Some(identity_definition()),
            });
            arrows.push(RawTheoryArrow {
                name: with_unit_name,
                type_maps: (term_to_hexpr(&suffix), with_left_unit_map(suffix.clone())?),
                definition: Some(with_left_unit_definition()?),
            });
        }
    }
    Ok(arrows)
}

fn partial_source_map(
    source: &SyntaxTerm,
    context: usize,
    applied: usize,
) -> Result<Hexpr, ElaborateError> {
    let mut prefix = slice_term(source, 0, applied);
    debug_assert_eq!(prefix.sources.len(), context);
    // A use of `partial.f.N` receives metavariable witnesses first, followed
    // by the captured prefix. Copy the witnesses because the target closure
    // type and the captured call to `f` both depend on them.
    prefix.targets = prefix
        .sources
        .iter()
        .copied()
        .chain(prefix.targets)
        .collect();
    Ok(term_to_hexpr(&prefix))
}

fn partial_target_map(
    source: &SyntaxTerm,
    target: &SyntaxTerm,
    context: usize,
    arity: usize,
    applied: usize,
    target_arity: usize,
) -> Result<Hexpr, ElaborateError> {
    let mut term = packed_slice_term(source, applied, arity);
    let target = packed_slice_term(target, 0, target_arity);
    let (target_sources, target_targets) = term.append(target);
    assert_eq!(
        term.sources.len(),
        context,
        "source type-map context should retain its original arity"
    );
    assert_eq!(
        target_sources.len(),
        context,
        "source and target type maps should have matching contexts"
    );
    for (source, target_source) in term.sources.clone().into_iter().zip(target_sources) {
        term.unify(source, target_source);
    }

    let [remaining] = term.targets.as_slice() else {
        unreachable!("a packed source slice should have one target");
    };
    let remaining = *remaining;
    let [target] = target_targets.as_slice() else {
        unreachable!("a packed target slice should have one target");
    };
    let target = *target;
    let function = add_graph_operation(&mut term, FN_HOM_TYPE, vec![remaining, target])?;
    term.targets = vec![function];
    term.quotient()
        .expect("unit-labeled syntax graphs should always quotient");
    Ok(term_to_hexpr(&term))
}

fn packed_slice_term(source: &SyntaxTerm, start: usize, end: usize) -> SyntaxTerm {
    let mut slice = slice_term(source, start, end);
    pack_targets(&mut slice);
    slice
}

fn slice_term(term: &SyntaxTerm, start: usize, end: usize) -> SyntaxTerm {
    let selected_targets = &term.targets[start..end];
    let mut retained_nodes = vec![false; term.hypergraph.nodes.len()];
    for node in &term.sources {
        retained_nodes[node.0] = true;
    }
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

    // A retained multi-output operation brings all of its incident nodes into
    // the subgraph, even when only some outputs are exposed at the boundary.
    for (index, boundary) in term.hypergraph.adjacency.iter().enumerate() {
        if retained_edges[index] {
            for node in boundary.sources.iter().chain(&boundary.targets) {
                retained_nodes[node.0] = true;
            }
        }
    }

    let mut slice = term.clone();
    slice.targets = selected_targets.to_vec();
    let discarded_edges = retained_edges
        .iter()
        .enumerate()
        .filter_map(|(index, retained)| (!retained).then_some(EdgeId(index)))
        .collect::<Vec<_>>();
    slice.delete_edges(&discarded_edges);
    let discarded_nodes = retained_nodes
        .iter()
        .enumerate()
        .filter_map(|(index, retained)| (!retained).then_some(NodeId(index)))
        .collect::<Vec<_>>();
    slice.delete_nodes(&discarded_nodes);
    slice
}

fn pack_targets(term: &mut SyntaxTerm) {
    let targets = term.targets.clone();
    let packed = pack_nodes(term, &targets, (), UNIT_TYPE, PRODUCT_TYPE);
    term.targets = vec![packed];
}

fn with_left_unit_map(mut remaining: SyntaxTerm) -> Result<Hexpr, ElaborateError> {
    let [payload] = remaining.targets.as_slice() else {
        unreachable!("a packed remaining-input map should have one target");
    };
    let payload = *payload;
    let unit = add_graph_operation(&mut remaining, UNIT_TYPE, vec![])?;
    let product = add_graph_operation(&mut remaining, PRODUCT_TYPE, vec![unit, payload])?;
    remaining.targets = vec![product];
    Ok(term_to_hexpr(&remaining))
}

/// Build the CMC term implementing `partial.f.N`.
///
/// Write the canonical domain of `f` as
/// `A_0 * (A_1 * ... * (A_(N-1) * R))`, where `R` contains the remaining
/// arguments. For the nontrivial case `0 < N < arity(f)`, repeatedly prepend
/// one deferred captured argument:
///
/// ```text
/// prepend_i =
///   (name.with-left-unit.D_i lift);
///   ((A_i defer) tensor (name.identity.D_i lift))
///
/// partial.f.N =
///   prepend_N; ...; prepend_1; (name.f lift)
/// ```
///
/// Each prepend step has the following type:
///
/// ```text
/// with-left-unit-D_i              : D_i       => 1 * D_i
/// defer(A_i) tensor identity-D_i  : 1 * D_i   => A_i * D_i
/// --------------------------------------------------------
/// prepend_i                       : D_i       => A_i * D_i
/// ```
///
/// This is the general form of the hand-written matrix construction using
/// `index-with-unit`, `input defer`, `index-id`, and `buf-at`.
fn partial_definition(
    original: &RawTheoryArrow,
    context: usize,
    arity: usize,
    applied: usize,
    helper_names: &[(Operation, Operation)],
) -> Result<Hexpr, ElaborateError> {
    let mut term = OpenHypergraph::empty();
    let inputs = (0..context + applied)
        .map(|_| term.new_node(()))
        .collect::<Vec<_>>();
    term.sources = inputs.clone();
    let (context_inputs, captured_inputs) = inputs.split_at(context);

    // Reusing these context node IDs on each named operation is the graph-level
    // representation of copying the metavariable context.
    //
    //   context ──▶ name.f ──▶ function pointer
    //              ──▶ lift ──▶ (pack(A_0, ..., A_(k-1)) => B)
    let original_closure = add_named_closure(&mut term, &original.name, context_inputs)?;

    let body = if applied == 0 {
        // Nothing is captured: partial.f.0 is simply the lifted name of f.
        //
        //   context ──▶ name.f ──▶ lift
        //              ──▶ (pack(A_0, ..., A_(k-1)) => B)
        original_closure
    } else {
        if applied == arity {
            // Pack the full supplied domain and capture it as one closure.
            let captured_value = pack_value_nodes(&mut term, captured_inputs);
            let captured = add_graph_operation(&mut term, DEFER, vec![captured_value])?;
            // There is no remaining R. Compose 1 => P with P => B directly,
            // producing the fully captured closure 1 => B.
            //
            //   (1 => P) ─┐
            //             ├─▶ compose ──▶ (1 => B)
            //   (P => B) ─┘
            add_graph_operation(&mut term, COMPOSE, vec![captured, original_closure])?
        } else {
            debug_assert_eq!(captured_inputs.len(), helper_names.len());
            let mut prepare = None;
            for (captured_input, (identity_name, with_unit_name)) in
                captured_inputs.iter().zip(helper_names).rev()
            {
                let captured = add_graph_operation(&mut term, DEFER, vec![*captured_input])?;
                let identity_closure = add_named_closure(&mut term, identity_name, context_inputs)?;
                let with_unit_closure =
                    add_named_closure(&mut term, with_unit_name, context_inputs)?;
                let append_capture =
                    add_graph_operation(&mut term, TENSOR, vec![captured, identity_closure])?;
                let prepend = add_graph_operation(
                    &mut term,
                    COMPOSE,
                    vec![with_unit_closure, append_capture],
                )?;
                prepare = Some(match prepare {
                    None => prepend,
                    Some(previous) => {
                        add_graph_operation(&mut term, COMPOSE, vec![previous, prepend])?
                    }
                });
            }
            let prepare = prepare.expect("a nontrivial partial captures at least one input");

            // Finally compose R => pack(A_1, ..., A_k) with the lifted
            // original function.
            //
            //   (R => pack(A_0, ..., A_(k-1))) ─┐
            //                                   ├─▶ compose ──▶ (R => B)
            //   (pack(A_0, ..., A_(k-1)) => B) ─┘
            add_graph_operation(&mut term, COMPOSE, vec![prepare, original_closure])?
        }
    };
    term.targets = vec![body];
    Ok(term_to_hexpr(&term))
}

fn add_named_closure(
    term: &mut OpenHypergraph<(), Operation>,
    name: &Operation,
    context_inputs: &[NodeId],
) -> Result<NodeId, ElaborateError> {
    //   context... ──▶ name.<operation> ──▶ pointer ──▶ lift ──▶ closure
    let pointer = add_graph_operation(
        term,
        &format!("{NAME_PREFIX}{name}"),
        context_inputs.to_vec(),
    )?;
    add_graph_operation(term, LIFT, vec![pointer])
}

fn pack_value_nodes(term: &mut OpenHypergraph<(), Operation>, values: &[NodeId]) -> NodeId {
    pack_nodes(term, values, (), UNIT_INTRO, PRODUCT_INTRO)
}

fn add_graph_operation(
    term: &mut OpenHypergraph<(), Operation>,
    name: &str,
    sources: Vec<NodeId>,
) -> Result<NodeId, ElaborateError> {
    //   source 0 ─┐
    //   source 1 ─┼─▶ operation ──▶ fresh target
    //   ...      ─┘
    let target = term.new_node(());
    term.new_edge(parse_op(name)?, (sources, vec![target]));
    Ok(target)
}

fn identity_definition() -> Hexpr {
    //   R ─────────────────▶ R
    let mut term = OpenHypergraph::empty();
    let wire = term.new_node(());
    term.sources = vec![wire];
    term.targets = vec![wire];
    term_to_hexpr(&term)
}

fn with_left_unit_definition() -> Result<Hexpr, ElaborateError> {
    //   ∅ ──▶ unit.intro ──▶ 1 ─┐
    //                           ├─▶ *.intro ──▶ 1 * R
    //   R ──────────────────────┘
    let mut term = OpenHypergraph::empty();
    let wire = term.new_node(());
    term.sources = vec![wire];
    let unit = add_graph_operation(&mut term, UNIT_INTRO, vec![])?;
    let product = add_graph_operation(&mut term, PRODUCT_INTRO, vec![unit, wire])?;
    term.targets = vec![product];
    Ok(term_to_hexpr(&term))
}

fn generated_helper_name(kind: &str, partial: &Operation) -> Result<Operation, ElaborateError> {
    format!("{GENERATED_PARTIAL_PREFIX}{kind}.{}", partial.as_str())
        .parse()
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(partial.to_string()))
}

fn parse_op(name: &str) -> Result<Operation, ElaborateError> {
    name.parse()
        .map_err(|_| ElaborateError::InvalidGeneratedOperation(name.to_string()))
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
        for helper in [
            "catena.partial.identity.0.partial.f.2",
            "catena.partial.with-left-unit.0.partial.f.2",
            "catena.partial.identity.1.partial.f.2",
            "catena.partial.with-left-unit.1.partial.f.2",
        ] {
            let helper: hexpr::Operation = helper.parse().unwrap();
            assert!(
                elaborated.theories[&program].arrows.contains_key(&helper),
                "missing suffix-specialized helper {helper}"
            );
        }
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
