//! Inline named evaluations whose callee has closures on its interface.
//!
//! Direct calls to closure-boundary definitions are expanded before
//! `forget_closures`. A lifted function name becomes a `name.f -> eval` pair
//! only while closures are being forgotten, so it needs a second, graph-level
//! specialization step. At this point closure arguments are delimited by
//! `ClosureMarker` edges; opening those adapters and splicing the forgotten
//! callee body connects the caller and callee control-flow regions.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryId, TheorySet},
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
        FN_HOM_TYPE, PRODUCT_ELIM, PRODUCT_INTRO, PRODUCT_TYPE, UNIT_ELIM, UNIT_INTRO, UNIT_TYPE,
    },
};

#[derive(Debug, Error)]
pub enum InlineNamedError {
    #[error("missing theory `{0}` while specializing named closure evaluations")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing forgotten body for closure-boundary definition `{theory}.{definition}`")]
    MissingBody { theory: String, definition: String },
    #[error(
        "cannot open argument node w{node} while specializing `name.{definition}` in `{theory}.{caller}`"
    )]
    CannotOpenArgument {
        theory: String,
        caller: String,
        definition: String,
        node: usize,
    },
    #[error(
        "forgotten boundary mismatch while specializing `name.{definition}` in `{theory}.{caller}`: expected {expected} inputs and {expected_targets} outputs, found {actual} inputs and {actual_targets} outputs"
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
    #[error(
        "closure-boundary named evaluation `{operation}` remains after specialization in `{theory}.{definition}`"
    )]
    RemainingNamedEvaluation {
        theory: String,
        definition: String,
        operation: String,
    },
}

/// Inline every direct `name.f -> eval` pair for which `f` has a closure on
/// its source-level interface, then omit those template-only definitions from
/// the runtime graph set.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
    templates: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, InlineNamedError> {
    let mut output = forgotten.clone();

    for (theory_id, definitions) in &mut output {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| InlineNamedError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(InlineNamedError::NotUserTheory(theory_id.to_string()));
        };
        let theory_templates = templates
            .get(theory_id)
            .ok_or_else(|| InlineNamedError::MissingTheory(theory_id.to_string()))?;

        for (caller, term) in definitions.iter_mut() {
            loop {
                let Some(pair) = next_pair(term, arrows) else {
                    break;
                };
                let template = theory_templates.get(&pair.definition).ok_or_else(|| {
                    InlineNamedError::MissingBody {
                        theory: theory_id.to_string(),
                        definition: pair.definition.to_string(),
                    }
                })?;
                inline_pair(theory_id, caller, term, template, pair)?;
            }
        }

        validate_no_named_closure_evals(theory_id, definitions, arrows)?;

        // Closure-boundary definitions retained by early inlining exist only
        // because a `name.f` reference needed their body. All such references
        // have now been specialized into callers, so these are not runtime
        // functions and must not enter region conversion or code generation.
        definitions.retain(|definition, _| {
            arrows
                .get(definition)
                .is_none_or(|arrow| !arrow_has_closure_boundary(arrow))
        });
    }

    Ok(output)
}

/// Definitions with closure-bearing interfaces are retained by the early
/// inliner only when a first-class `name.f` reference still needs them. They
/// are post-forget specialization templates, never runtime entrypoints.
pub fn template_definitions(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, InlineNamedError> {
    let mut output = BTreeMap::new();
    for (theory_id, definitions) in forgotten {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| InlineNamedError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(InlineNamedError::NotUserTheory(theory_id.to_string()));
        };
        let selected = definitions
            .iter()
            .filter(|(definition, _)| {
                arrows
                    .get(*definition)
                    .is_some_and(arrow_has_closure_boundary)
            })
            .map(|(definition, term)| (definition.clone(), term.clone()))
            .collect::<BTreeMap<_, _>>();
        if !selected.is_empty() {
            output.insert(theory_id.clone(), selected);
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct NamedEvalPair {
    name: EdgeId,
    eval: EdgeId,
    definition: Operation,
}

fn next_pair(
    term: &ClosureForgottenTerm,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
) -> Option<NamedEvalPair> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .find_map(|(index, operation)| {
            let ClosureForgotten::Operation(operation) = operation else {
                return None;
            };
            if operation.as_str() != "eval" {
                return None;
            }
            let eval = EdgeId(index);
            let pointer = *term.hypergraph.adjacency[index].sources.get(1)?;
            let (name, name_operation) = producer_edges(term, pointer).find_map(|producer| {
                let ClosureForgotten::Operation(operation) = &term.hypergraph.edges[producer.0]
                else {
                    return None;
                };
                operation
                    .as_str()
                    .strip_prefix(NAME_PREFIX)
                    .map(|name| (producer, name))
            })?;
            let definition: Operation = name_operation.parse().ok()?;
            arrows
                .get(&definition)
                .is_some_and(arrow_has_closure_boundary)
                .then_some(NamedEvalPair {
                    name,
                    eval,
                    definition,
                })
        })
}

fn inline_pair(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    template: &ClosureForgottenTerm,
    pair: NamedEvalPair,
) -> Result<(), InlineNamedError> {
    let context = term.hypergraph.adjacency[pair.name.0]
        .sources
        .iter()
        .map(|source| term.hypergraph.nodes[source.0].clone())
        .collect::<Vec<_>>();
    let template = template
        .clone()
        .map_nodes(|object| instantiate_object(&object, &context));
    let eval_boundary = term.hypergraph.adjacency[pair.eval.0].clone();
    let Some(&packed_domain) = eval_boundary.sources.first() else {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &pair.definition,
            &template,
            0,
            eval_boundary.targets.len(),
        ));
    };

    let mut adapter_edges = BTreeSet::new();
    let mut visiting = HashSet::new();
    let inputs = open_argument(
        theory_id,
        caller,
        &pair.definition,
        term,
        packed_domain,
        &mut adapter_edges,
        &mut visiting,
    )?;

    if inputs.len() != template.sources.len()
        || eval_boundary.targets.len() != template.targets.len()
        || !same_types(term, &inputs, &template, &template.sources)
        || !same_types(term, &eval_boundary.targets, &template, &template.targets)
    {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &pair.definition,
            &template,
            inputs.len(),
            eval_boundary.targets.len(),
        ));
    }

    let mut deleted = adapter_edges.into_iter().map(EdgeId).collect::<Vec<_>>();
    deleted.push(pair.name);
    deleted.push(pair.eval);
    deleted.sort_by_key(|edge| edge.0);
    deleted.dedup();
    term.delete_edges(&deleted);

    let (template_sources, template_targets) = term.append(template);
    for (outer, inner) in inputs.into_iter().zip(template_sources) {
        term.unify(outer, inner);
    }
    for (inner, outer) in template_targets
        .into_iter()
        .zip(eval_boundary.targets.into_iter())
    {
        term.unify(inner, outer);
    }
    term.quotient().ok();
    Ok(())
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

/// Open the adapters introduced for an operation argument by
/// `forget_closures`. Products are unpacked recursively, closure markers are
/// replaced by their domain/codomain endpoints, and units disappear.
fn open_argument(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    term: &mut ClosureForgottenTerm,
    node: NodeId,
    deleted_edges: &mut BTreeSet<usize>,
    visiting: &mut HashSet<NodeId>,
) -> Result<Vec<NodeId>, InlineNamedError> {
    if !visiting.insert(node) {
        return Err(cannot_open(theory_id, caller, definition, node));
    }

    let ty = term.hypergraph.nodes[node.0].clone();
    let result = match &ty {
        Tree::Node(operation, _, children)
            if operation.as_str() == UNIT_TYPE && children.is_empty() =>
        {
            for edge in incident_structural_edges(term, node, &[UNIT_INTRO, UNIT_ELIM]) {
                deleted_edges.insert(edge.0);
            }
            Vec::new()
        }
        Tree::Node(operation, _, _) if operation.as_str() == PRODUCT_TYPE => {
            let producer = producer_edges(term, node)
                .find(|edge| is_structural(term, *edge, &[PRODUCT_INTRO, PRODUCT_ELIM]));
            let sources = if let Some(edge) = producer {
                deleted_edges.insert(edge.0);
                term.hypergraph.adjacency[edge.0].sources.clone()
            } else {
                // A closure domain is contravariant. When its product endpoint
                // has no producer, the inlined callee will produce the flattened
                // components; pack those components into the product consumed
                // by the captured closure body.
                let unflattener = to_unflattener(&ty).map_edges(ClosureForgotten::Operation);
                let (sources, targets) = term.append(unflattener);
                let [target] = targets.as_slice() else {
                    unreachable!("one-object unflattener should have one target")
                };
                term.unify(node, *target);
                sources
            };
            let mut opened = Vec::new();
            for source in sources {
                opened.extend(open_argument(
                    theory_id,
                    caller,
                    definition,
                    term,
                    source,
                    deleted_edges,
                    visiting,
                )?);
            }
            opened
        }
        Tree::Node(operation, _, _) if operation.as_str() == FN_HOM_TYPE => {
            let edge = producer_edges(term, node)
                .filter(|edge| {
                    matches!(
                        term.hypergraph.edges[edge.0],
                        ClosureForgotten::ClosureMarker
                    )
                })
                .next()
                .ok_or_else(|| cannot_open(theory_id, caller, definition, node))?;
            deleted_edges.insert(edge.0);
            let sources = term.hypergraph.adjacency[edge.0].sources.clone();
            let mut opened = Vec::new();
            for source in sources {
                opened.extend(open_argument(
                    theory_id,
                    caller,
                    definition,
                    term,
                    source,
                    deleted_edges,
                    visiting,
                )?);
            }
            opened
        }
        _ => vec![node],
    };
    visiting.remove(&node);
    Ok(result)
}

fn producer_edges(term: &ClosureForgottenTerm, node: NodeId) -> impl Iterator<Item = EdgeId> + '_ {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(move |(_, boundary)| boundary.targets.contains(&node))
        .map(|(index, _)| EdgeId(index))
}

fn incident_structural_edges(
    term: &ClosureForgottenTerm,
    node: NodeId,
    operations: &[&str],
) -> Vec<EdgeId> {
    term.hypergraph
        .adjacency
        .iter()
        .enumerate()
        .filter(|(_, boundary)| {
            boundary.sources.contains(&node) || boundary.targets.contains(&node)
        })
        .map(|(index, _)| EdgeId(index))
        .filter(|edge| is_structural(term, *edge, operations))
        .collect()
}

fn is_structural(term: &ClosureForgottenTerm, edge: EdgeId, operations: &[&str]) -> bool {
    matches!(
        &term.hypergraph.edges[edge.0],
        ClosureForgotten::Operation(operation)
            if operations.contains(&operation.as_str())
    )
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

fn arrow_has_closure_boundary(arrow: &metacat::theory::TheoryArrow) -> bool {
    contains_closure(&arrow.type_maps.0) || contains_closure(&arrow.type_maps.1)
}

fn contains_closure(type_map: &metacat::theory::Term) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|operation| operation.as_str() == FN_HOM_TYPE)
}

fn validate_no_named_closure_evals(
    theory_id: &TheoryId,
    definitions: &BTreeMap<Operation, ClosureForgottenTerm>,
    arrows: &BTreeMap<Operation, metacat::theory::TheoryArrow>,
) -> Result<(), InlineNamedError> {
    for (definition, term) in definitions {
        if let Some(pair) = next_pair(term, arrows) {
            return Err(InlineNamedError::RemainingNamedEvaluation {
                theory: theory_id.to_string(),
                definition: definition.to_string(),
                operation: format!("name.{}", pair.definition),
            });
        }
    }
    Ok(())
}

fn cannot_open(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    node: NodeId,
) -> InlineNamedError {
    InlineNamedError::CannotOpenArgument {
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
    template: &ClosureForgottenTerm,
    actual: usize,
    actual_targets: usize,
) -> InlineNamedError {
    InlineNamedError::BoundaryMismatch {
        theory: theory_id.to_string(),
        caller: caller.to_string(),
        definition: definition.to_string(),
        expected: template.sources.len(),
        actual,
        expected_targets: template.targets.len(),
        actual_targets,
    }
}
