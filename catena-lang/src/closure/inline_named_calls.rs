//! Inline named calls that cannot keep their closure-bearing ABI at runtime.
//!
//! Ordinary closure forgetting turns `name.f ; lift` into a graph containing
//! `name.f`, the argument adapters for `eval`, and `eval` itself. This module
//! runs before closure-region conversion and handles that graph in four steps:
//!
//! 1. find an `eval` whose function pointer is produced by `name.f`;
//! 2. open the eval adapters to recover the fully forgotten call boundary;
//! 3. instantiate the already-forgotten body of `f` at that boundary;
//! 4. replace the complete name/adapter/eval fragment with that body.
//!
//! Spliced bodies may contain further named calls, so each definition is
//! processed until no closure-bearing named call remains.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryArrow, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::{
    nonstrict::to_unflattener,
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
    prefixes::NAME_PREFIX,
    report::TheoryTermMap,
    stdlib::constants::{
        EVAL, FN_HOM_TYPE, PRODUCT_ELIM, PRODUCT_INTRO, PRODUCT_TYPE, UNIT_ELIM, UNIT_INTRO,
        UNIT_TYPE,
    },
};

#[derive(Debug, Error)]
pub enum InlineNamedCallsError {
    #[error("missing theory `{0}` while inlining named calls")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing forgotten body for `{theory}.{definition}`")]
    MissingBody { theory: String, definition: String },
    #[error(
        "cannot open adapter node w{node} while inlining `name.{definition}` in `{theory}.{caller}`"
    )]
    CannotOpenAdapter {
        theory: String,
        caller: String,
        definition: String,
        node: usize,
    },
    #[error(
        "forgotten boundary mismatch while inlining `name.{definition}` in `{theory}.{caller}`: expected {expected} inputs and {expected_targets} outputs, found {actual} inputs and {actual_targets} outputs"
    )]
    BoundaryMismatch {
        theory: String,
        caller: String,
        definition: String,
        expected: usize,
        actual: usize,
        expected_targets: usize,
        actual_targets: usize,
    },
}

/// Inline every named call whose definition has a closure somewhere on
/// its source-level boundary. Such definitions are then removed because their
/// closure-bearing interfaces are not runtime ABIs.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, InlineNamedCallsError> {
    let mut output = forgotten.clone();

    for (theory_id, definitions) in &mut output {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| InlineNamedCallsError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(InlineNamedCallsError::NotUserTheory(theory_id.to_string()));
        };
        // Always use the immutable forgotten definitions as templates. This
        // makes recursive substitution independent of map iteration order.
        let templates = &forgotten[theory_id];

        for (caller, term) in definitions.iter_mut() {
            inline_calls_in_definition(theory_id, caller, term, arrows, templates)?;
        }

        remove_template_only_definitions(definitions, arrows);
    }

    Ok(output)
}

#[derive(Debug, Clone)]
struct NamedCall {
    name: EdgeId,
    eval: EdgeId,
    definition: Operation,
}

fn inline_calls_in_definition(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    templates: &BTreeMap<Operation, ClosureForgottenTerm>,
) -> Result<(), InlineNamedCallsError> {
    // Restart after every splice: appended templates may contain more calls,
    // while quotienting renumbers all nodes and edges.
    while let Some(call) = find_next_call(term, arrows) {
        let template =
            templates
                .get(&call.definition)
                .ok_or_else(|| InlineNamedCallsError::MissingBody {
                    theory: theory_id.to_string(),
                    definition: call.definition.to_string(),
                })?;
        inline_call(theory_id, caller, term, template, call)?;
    }
    Ok(())
}

/// Find the ordinary post-forget shape `name.f -> eval`. First-order calls are
/// left alone; only calls whose original definition boundary contains a
/// closure need to lose their call adapter before region conversion.
fn find_next_call(
    term: &ClosureForgottenTerm,
    arrows: &BTreeMap<Operation, TheoryArrow>,
) -> Option<NamedCall> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .find_map(|(index, operation)| {
            let ClosureForgotten::Operation(operation) = operation else {
                return None;
            };
            if operation.as_str() != EVAL {
                return None;
            }

            let eval = EdgeId(index);
            let pointer = *term.hypergraph.adjacency[index].sources.get(1)?;
            let (name, definition) = producer_edges(term, pointer).find_map(|producer| {
                let ClosureForgotten::Operation(operation) = &term.hypergraph.edges[producer.0]
                else {
                    return None;
                };
                let definition = operation.as_str().strip_prefix(NAME_PREFIX)?.parse().ok()?;
                Some((producer, definition))
            })?;

            arrows
                .get(&definition)
                .is_some_and(arrow_has_closure_boundary)
                .then_some(NamedCall {
                    name,
                    eval,
                    definition,
                })
        })
}

fn inline_call(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    template: &ClosureForgottenTerm,
    call: NamedCall,
) -> Result<(), InlineNamedCallsError> {
    let template = instantiate_template(term, template, call.name);
    let boundary = recover_call_boundary(theory_id, caller, term, &call)?;
    verify_boundary(theory_id, caller, term, &call, &boundary, &template)?;
    replace_call(term, call, boundary, template);
    Ok(())
}

/// Type variables used by `f` are supplied on the inputs of `name.f`.
fn instantiate_template(
    term: &ClosureForgottenTerm,
    template: &ClosureForgottenTerm,
    name: EdgeId,
) -> ClosureForgottenTerm {
    let context = term.hypergraph.adjacency[name.0]
        .sources
        .iter()
        .map(|source| term.hypergraph.nodes[source.0].clone())
        .collect::<Vec<_>>();
    template
        .clone()
        .map_nodes(|object| instantiate_object(&object, &context))
}

#[derive(Debug)]
struct CallBoundary {
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
    adapter_edges: BTreeSet<usize>,
}

/// Walk only the structural adapters adjacent to `eval`. The resulting inputs
/// and outputs are the flattened boundary at which the forgotten body of `f`
/// can be spliced.
fn recover_call_boundary(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    call: &NamedCall,
) -> Result<CallBoundary, InlineNamedCallsError> {
    let eval = term.hypergraph.adjacency[call.eval.0].clone();
    let Some(&domain) = eval.sources.first() else {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &call.definition,
            0,
            eval.targets.len(),
            0,
            0,
        ));
    };

    let mut adapter_edges = BTreeSet::new();
    let mut visiting = HashSet::new();
    let inputs = open_input_adapter(
        theory_id,
        caller,
        &call.definition,
        term,
        domain,
        &mut adapter_edges,
        &mut visiting,
    )?;

    let mut outputs = Vec::new();
    for target in eval.targets {
        outputs.extend(open_output_adapter(
            theory_id,
            caller,
            &call.definition,
            term,
            target,
            &mut adapter_edges,
            &mut visiting,
        )?);
    }

    Ok(CallBoundary {
        inputs,
        outputs,
        adapter_edges,
    })
}

/// Reverse the source adapter created for a non-CMC operation.
fn open_input_adapter(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    term: &mut ClosureForgottenTerm,
    node: NodeId,
    adapter_edges: &mut BTreeSet<usize>,
    visiting: &mut HashSet<NodeId>,
) -> Result<Vec<NodeId>, InlineNamedCallsError> {
    begin_node(theory_id, caller, definition, node, visiting)?;
    let ty = term.hypergraph.nodes[node.0].clone();

    let opened = if is_unit(&ty) {
        let edge = incident_edges(term, node)
            .find(|edge| is_unit_adapter(term, *edge))
            .ok_or_else(|| cannot_open(theory_id, caller, definition, node))?;
        adapter_edges.insert(edge.0);
        Vec::new()
    } else if is_product(&ty) {
        let components = if let Some(edge) =
            producer_edges(term, node).find(|edge| is_product_adapter(term, *edge))
        {
            adapter_edges.insert(edge.0);
            term.hypergraph.adjacency[edge.0].sources.clone()
        } else {
            // Contravariant closure domains point away from their flattened
            // components. Add the inverse structural adapter so the forgotten
            // callee can connect at the same flat boundary.
            let adapter = to_unflattener(&ty).map_edges(ClosureForgotten::Operation);
            let (sources, targets) = term.append(adapter);
            let [target] = targets.as_slice() else {
                unreachable!("one product unflattener should have one target")
            };
            term.unify(node, *target);
            sources
        };
        open_input_components(
            theory_id,
            caller,
            definition,
            term,
            components,
            adapter_edges,
            visiting,
        )?
    } else if is_closure(&ty) {
        let marker = producer_edges(term, node)
            .find(|edge| {
                matches!(
                    term.hypergraph.edges[edge.0],
                    ClosureForgotten::ClosureMarker
                )
            })
            .ok_or_else(|| cannot_open(theory_id, caller, definition, node))?;
        adapter_edges.insert(marker.0);
        let components = term.hypergraph.adjacency[marker.0].sources.clone();
        open_input_components(
            theory_id,
            caller,
            definition,
            term,
            components,
            adapter_edges,
            visiting,
        )?
    } else {
        vec![node]
    };

    visiting.remove(&node);
    Ok(opened)
}

fn open_input_components(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    term: &mut ClosureForgottenTerm,
    components: Vec<NodeId>,
    adapter_edges: &mut BTreeSet<usize>,
    visiting: &mut HashSet<NodeId>,
) -> Result<Vec<NodeId>, InlineNamedCallsError> {
    let mut opened = Vec::new();
    for component in components {
        opened.extend(open_input_adapter(
            theory_id,
            caller,
            definition,
            term,
            component,
            adapter_edges,
            visiting,
        )?);
    }
    Ok(opened)
}

/// Follow the target flatteners created for a non-CMC operation.
fn open_output_adapter(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    term: &mut ClosureForgottenTerm,
    node: NodeId,
    adapter_edges: &mut BTreeSet<usize>,
    visiting: &mut HashSet<NodeId>,
) -> Result<Vec<NodeId>, InlineNamedCallsError> {
    begin_node(theory_id, caller, definition, node, visiting)?;
    let ty = term.hypergraph.nodes[node.0].clone();

    let opened = if is_unit(&ty) {
        let edge = incident_edges(term, node)
            .find(|edge| is_unit_adapter(term, *edge))
            .ok_or_else(|| cannot_open(theory_id, caller, definition, node))?;
        adapter_edges.insert(edge.0);
        Vec::new()
    } else if is_product(&ty) {
        let edge = consumer_edges(term, node)
            .find(|edge| is_product_adapter(term, *edge))
            .ok_or_else(|| cannot_open(theory_id, caller, definition, node))?;
        adapter_edges.insert(edge.0);
        let components = term.hypergraph.adjacency[edge.0].targets.clone();
        open_output_components(
            theory_id,
            caller,
            definition,
            term,
            components,
            adapter_edges,
            visiting,
        )?
    } else {
        vec![node]
    };

    visiting.remove(&node);
    Ok(opened)
}

fn open_output_components(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    term: &mut ClosureForgottenTerm,
    components: Vec<NodeId>,
    adapter_edges: &mut BTreeSet<usize>,
    visiting: &mut HashSet<NodeId>,
) -> Result<Vec<NodeId>, InlineNamedCallsError> {
    let mut opened = Vec::new();
    for component in components {
        opened.extend(open_output_adapter(
            theory_id,
            caller,
            definition,
            term,
            component,
            adapter_edges,
            visiting,
        )?);
    }
    Ok(opened)
}

fn begin_node(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    node: NodeId,
    visiting: &mut HashSet<NodeId>,
) -> Result<(), InlineNamedCallsError> {
    visiting
        .insert(node)
        .then_some(())
        .ok_or_else(|| cannot_open(theory_id, caller, definition, node))
}

fn verify_boundary(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &ClosureForgottenTerm,
    call: &NamedCall,
    boundary: &CallBoundary,
    template: &ClosureForgottenTerm,
) -> Result<(), InlineNamedCallsError> {
    if boundary.inputs.len() == template.sources.len()
        && boundary.outputs.len() == template.targets.len()
        && same_types(term, &boundary.inputs, template, &template.sources)
        && same_types(term, &boundary.outputs, template, &template.targets)
    {
        return Ok(());
    }

    Err(boundary_mismatch(
        theory_id,
        caller,
        &call.definition,
        boundary.inputs.len(),
        boundary.outputs.len(),
        template.sources.len(),
        template.targets.len(),
    ))
}

/// Replace one forgotten named call with an instantiated copy of its definition.
///
/// Forgetting closures turns a source-level named closure call into a graph
/// containing the function name, `eval`, and product/unit adapters:
///
/// ```text
/// context ───────────────────────► name.f ──► function pointer ─┐
///                                                               ▼
/// flattened inputs ──► input adapters ──► packed input ──► eval
///                                                               │
///                                                               ▼
/// flattened outputs ◄── output adapters ◄── packed output ─────┘
/// ```
///
/// `recover_call_boundary` has already opened the adapters and returned the
/// flattened inputs and outputs. This function removes the complete call
/// fragment and splices the forgotten body of `f` directly at that boundary:
///
/// ```text
/// flattened inputs ──► [ instantiated body of f ] ──► flattened outputs
/// ```
fn replace_call(
    term: &mut ClosureForgottenTerm,
    call: NamedCall,
    boundary: CallBoundary,
    template: ClosureForgottenTerm,
) {
    let CallBoundary {
        inputs,
        outputs,
        adapter_edges,
    } = boundary;
    let mut deleted = adapter_edges.into_iter().map(EdgeId).collect::<Vec<_>>();
    deleted.extend([call.name, call.eval]);
    deleted.sort_by_key(|edge| edge.0);
    deleted.dedup();

    // Collect the edges forming the old `name.f -> eval` call and its adapters.
    let deleted_edges = deleted.iter().map(|edge| edge.0).collect::<BTreeSet<_>>();

    // A node must survive if it is part of the enclosing definition, is one of
    // the recovered splice boundaries, or is incident to an edge we keep.
    let retained_nodes = term
        .sources
        .iter()
        .chain(&term.targets)
        .chain(&inputs)
        .chain(&outputs)
        .chain(
            term.hypergraph
                .adjacency
                .iter()
                .enumerate()
                .filter(|(index, _)| !deleted_edges.contains(index))
                .flat_map(|(_, edge)| edge.sources.iter().chain(&edge.targets)),
        )
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();

    // Among nodes touched by the removed edges, delete only those private to
    // the fragment. These are typically the function pointer produced by
    // `name.f` and the intermediate packed product/unit adapter nodes.
    let deleted_nodes = deleted
        .iter()
        .flat_map(|edge| {
            let adjacency = &term.hypergraph.adjacency[edge.0];
            adjacency.sources.iter().chain(&adjacency.targets)
        })
        .filter(|node| !retained_nodes.contains(&node.0))
        .map(|node| node.0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(NodeId)
        .collect::<Vec<_>>();

    // Removing hyperedges does not automatically remove their nodes, so delete
    // the fragment-private nodes explicitly. Node deletion compacts the node
    // array; use its witness map to translate boundaries saved with the old IDs.
    term.delete_edges(&deleted);
    let node_map = term.hypergraph.delete_nodes_witness(&deleted_nodes);
    let remap = |node: NodeId| {
        NodeId(node_map[node.0].expect("retained call boundary node should survive deletion"))
    };
    term.sources
        .iter_mut()
        .for_each(|node| *node = remap(*node));
    term.targets
        .iter_mut()
        .for_each(|node| *node = remap(*node));
    let inputs = inputs.into_iter().map(remap).collect::<Vec<_>>();
    let outputs = outputs.into_iter().map(remap).collect::<Vec<_>>();

    // Append the instantiated body and identify its external boundary with the
    // remapped boundary of the removed call.
    let (template_sources, template_targets) = term.append(template);
    for (outer, inner) in inputs.into_iter().zip(template_sources) {
        term.unify(outer, inner);
    }
    for (inner, outer) in template_targets.into_iter().zip(outputs) {
        term.unify(inner, outer);
    }
    term.quotient().ok();
}

fn instantiate_object(
    object: &Tree<(), Operation>,
    context: &[Tree<(), Operation>],
) -> Tree<(), Operation> {
    match object {
        Tree::Empty => Tree::Empty,
        Tree::Leaf(index, ()) => context
            .get(*index)
            .cloned()
            .unwrap_or_else(|| object.clone()),
        Tree::Node(operation, arity, children) => Tree::Node(
            operation.clone(),
            *arity,
            children
                .iter()
                .map(|child| instantiate_object(child, context))
                .collect(),
        ),
    }
}

fn producer_edges(term: &ClosureForgottenTerm, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(move |(_, boundary)| boundary.targets.contains(&node))
        .map(|(index, _)| EdgeId(index))
}

fn consumer_edges(term: &ClosureForgottenTerm, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(move |(_, boundary)| boundary.sources.contains(&node))
        .map(|(index, _)| EdgeId(index))
}

fn incident_edges(term: &ClosureForgottenTerm, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(move |(_, boundary)| {
            boundary.sources.contains(&node) || boundary.targets.contains(&node)
        })
        .map(|(index, _)| EdgeId(index))
}

fn is_product_adapter(term: &ClosureForgottenTerm, edge: EdgeId) -> bool {
    is_operation(term, edge, &[PRODUCT_INTRO, PRODUCT_ELIM])
}

fn is_unit_adapter(term: &ClosureForgottenTerm, edge: EdgeId) -> bool {
    is_operation(term, edge, &[UNIT_INTRO, UNIT_ELIM])
}

fn is_operation(term: &ClosureForgottenTerm, edge: EdgeId, names: &[&str]) -> bool {
    matches!(
        &term.hypergraph.edges[edge.0],
        ClosureForgotten::Operation(operation) if names.contains(&operation.as_str())
    )
}

fn is_product(object: &Tree<(), Operation>) -> bool {
    matches!(object, Tree::Node(operation, _, _) if operation.as_str() == PRODUCT_TYPE)
}

fn is_unit(object: &Tree<(), Operation>) -> bool {
    matches!(
        object,
        Tree::Node(operation, _, children)
            if operation.as_str() == UNIT_TYPE && children.is_empty()
    )
}

fn is_closure(object: &Tree<(), Operation>) -> bool {
    matches!(object, Tree::Node(operation, _, _) if operation.as_str() == FN_HOM_TYPE)
}

fn same_types(
    outer: &ClosureForgottenTerm,
    outer_nodes: &[NodeId],
    inner: &ClosureForgottenTerm,
    inner_nodes: &[NodeId],
) -> bool {
    outer_nodes
        .iter()
        .zip(inner_nodes)
        .all(|(outer_node, inner_node)| {
            outer.hypergraph.nodes[outer_node.0] == inner.hypergraph.nodes[inner_node.0]
        })
}

fn arrow_has_closure_boundary(arrow: &TheoryArrow) -> bool {
    contains_closure(&arrow.type_maps.0) || contains_closure(&arrow.type_maps.1)
}

fn contains_closure(type_map: &metacat::theory::Term) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|operation| operation.as_str() == FN_HOM_TYPE)
}

fn remove_template_only_definitions(
    definitions: &mut BTreeMap<Operation, ClosureForgottenTerm>,
    arrows: &BTreeMap<Operation, TheoryArrow>,
) {
    definitions.retain(|definition, _| {
        arrows
            .get(definition)
            .is_none_or(|arrow| !arrow_has_closure_boundary(arrow))
    });
}

fn cannot_open(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    node: NodeId,
) -> InlineNamedCallsError {
    InlineNamedCallsError::CannotOpenAdapter {
        theory: theory_id.to_string(),
        caller: caller.to_string(),
        definition: definition.to_string(),
        node: node.0,
    }
}

fn boundary_mismatch(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    actual: usize,
    actual_targets: usize,
    expected: usize,
    expected_targets: usize,
) -> InlineNamedCallsError {
    InlineNamedCallsError::BoundaryMismatch {
        theory: theory_id.to_string(),
        caller: caller.to_string(),
        definition: definition.to_string(),
        expected,
        actual,
        expected_targets,
        actual_targets,
    }
}

#[cfg(test)]
mod tests {
    use metacat::theory::{RawTheorySet, TheorySet};

    use crate::{
        pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
        prefixes::NAME_PREFIX,
    };

    #[test]
    fn replaces_the_complete_named_call_fragment_with_the_forgotten_body() {
        let source = r#"
            (def program apply-closure :
              ({1 (bool val)} =>) -> ({(bool val) (bool val)} *)
              = (run bool.copy *.intro))
            (def program use-named-closure :
              (bool val) -> ({(bool val) (bool val)} *)
              = ([captured.]
                  ([.captured] defer [inner.]
                    ({([.inner] defer) (name.apply-closure lift)} compose run))))
        "#;
        let raw = RawTheorySet::from_texts(crate::stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = crate::elaborate::elaborate(raw).expect("test theory should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theory should interpret");
        let types = crate::check::check(&theory_set).expect("test theory should check");
        let forgotten = crate::pass::forget_closures::run(&theory_set, &types)
            .expect("test theory should forget closures");
        let program = metacat::theory::TheoryId("program".parse().unwrap());
        let apply_closure: hexpr::Operation = "apply-closure".parse().unwrap();
        let use_named: hexpr::Operation = "use-named-closure".parse().unwrap();

        let before = &forgotten[&program][&use_named];
        assert!(before.hypergraph.edges.iter().any(|edge| {
            matches!(edge, ClosureForgotten::Operation(operation)
                if operation.as_str() == "eval")
        }));
        assert!(before.hypergraph.edges.iter().any(|edge| {
            matches!(edge, ClosureForgotten::Operation(operation)
                if operation.as_str() == format!("{NAME_PREFIX}{apply_closure}"))
        }));
        assert!(
            before
                .hypergraph
                .edges
                .iter()
                .any(|edge| matches!(edge, ClosureForgotten::ClosureMarker))
        );

        let specialized =
            super::run(&theory_set, &forgotten).expect("named closure-boundary call should inline");
        let after = &specialized[&program][&use_named];
        assert!(after.hypergraph.edges.iter().all(|edge| {
            !matches!(edge, ClosureForgotten::Operation(operation)
                if operation.as_str() == "eval"
                    || operation.as_str() == format!("{NAME_PREFIX}{apply_closure}"))
        }));
        assert!(!specialized[&program].contains_key(&apply_closure));
        assert_no_isolated_internal_nodes(after);
    }

    fn assert_no_isolated_internal_nodes(term: &ClosureForgottenTerm) {
        let mut referenced = vec![false; term.hypergraph.nodes.len()];
        for node in term.sources.iter().chain(&term.targets).chain(
            term.hypergraph
                .adjacency
                .iter()
                .flat_map(|edge| edge.sources.iter().chain(&edge.targets)),
        ) {
            referenced[node.0] = true;
        }
        assert!(
            referenced.into_iter().all(|referenced| referenced),
            "named-call inlining left isolated internal nodes"
        );
    }
}
