//! Specialize named evaluations before closure-region conversion.
//!
//! `forget_closures` preserves a statically known lifted call as one
//! [`ClosureForgotten::NamedEval`] edge whose boundary already matches the
//! forgotten boundary of the callee. Specialization is therefore ordinary
//! template splicing; it never has to reconstruct product or closure adapters.

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::{
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
    report::TheoryTermMap,
    stdlib::constants::FN_HOM_TYPE,
};

#[derive(Debug, Error)]
pub enum NamedEvalError {
    #[error("missing theory `{0}` while specializing named evaluations")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing forgotten body for named evaluation `{theory}.{definition}`")]
    MissingBody { theory: String, definition: String },
    #[error(
        "forgotten boundary mismatch for named evaluation `{theory}.{definition}` in `{caller}`: expected {expected} inputs and {expected_targets} outputs, found {actual} inputs and {actual_targets} outputs"
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

/// Inline every `NamedEval` and remove closure-boundary template definitions
/// from the runtime graph set.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
    templates: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, NamedEvalError> {
    let mut output = forgotten.clone();

    for (theory_id, definitions) in &mut output {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| NamedEvalError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(NamedEvalError::NotUserTheory(theory_id.to_string()));
        };
        let theory_templates = templates
            .get(theory_id)
            .ok_or_else(|| NamedEvalError::MissingTheory(theory_id.to_string()))?;

        for (caller, term) in definitions.iter_mut() {
            while let Some(named_eval) = next(term) {
                let template = theory_templates
                    .get(&named_eval.definition)
                    .ok_or_else(|| NamedEvalError::MissingBody {
                        theory: theory_id.to_string(),
                        definition: named_eval.definition.to_string(),
                    })?;
                inline(theory_id, caller, term, template, named_eval)?;
            }
        }

        definitions.retain(|definition, _| {
            arrows
                .get(definition)
                .is_none_or(|arrow| !arrow_has_closure_boundary(arrow))
        });
    }

    Ok(output)
}

/// Closure-boundary definitions retained by early inlining are templates for
/// `NamedEval`; they are not runtime functions.
pub fn templates(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, NamedEvalError> {
    let mut output = BTreeMap::new();
    for (theory_id, definitions) in forgotten {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| NamedEvalError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { arrows, .. } = theory else {
            return Err(NamedEvalError::NotUserTheory(theory_id.to_string()));
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
struct NamedEval {
    edge: EdgeId,
    definition: Operation,
    context: usize,
}

fn next(term: &ClosureForgottenTerm) -> Option<NamedEval> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .find_map(|(edge, operation)| match operation {
            ClosureForgotten::NamedEval {
                definition,
                context,
            } => Some(NamedEval {
                edge: EdgeId(edge),
                definition: definition.clone(),
                context: *context,
            }),
            ClosureForgotten::Operation(_) | ClosureForgotten::ClosureMarker => None,
        })
}

fn inline(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    template: &ClosureForgottenTerm,
    named_eval: NamedEval,
) -> Result<(), NamedEvalError> {
    let boundary = term.hypergraph.adjacency[named_eval.edge.0].clone();
    if boundary.sources.len() < named_eval.context {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &named_eval.definition,
            template,
            0,
            boundary.targets.len(),
        ));
    }
    let context = boundary.sources[..named_eval.context]
        .iter()
        .map(|source| term.hypergraph.nodes[source.0].clone())
        .collect::<Vec<_>>();
    let inputs = boundary.sources[named_eval.context..].to_vec();
    let template = template
        .clone()
        .map_nodes(|object| instantiate_object(&object, &context));

    if inputs.len() != template.sources.len()
        || boundary.targets.len() != template.targets.len()
        || !same_types(term, &inputs, &template, &template.sources)
        || !same_types(term, &boundary.targets, &template, &template.targets)
    {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &named_eval.definition,
            &template,
            inputs.len(),
            boundary.targets.len(),
        ));
    }

    term.delete_edges(&[named_eval.edge]);
    let (template_sources, template_targets) = term.append(template);
    for (outer, inner) in inputs.into_iter().zip(template_sources) {
        term.unify(outer, inner);
    }
    for (inner, outer) in template_targets.into_iter().zip(boundary.targets) {
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
    [&arrow.type_maps.0, &arrow.type_maps.1]
        .into_iter()
        .any(|type_map| {
            type_map
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation.as_str() == FN_HOM_TYPE)
        })
}

fn boundary_mismatch(
    theory_id: &TheoryId,
    caller: &Operation,
    definition: &Operation,
    template: &ClosureForgottenTerm,
    actual: usize,
    actual_targets: usize,
) -> NamedEvalError {
    NamedEvalError::BoundaryMismatch {
        theory: theory_id.to_string(),
        caller: caller.to_string(),
        definition: definition.to_string(),
        expected: template.sources.len(),
        actual,
        expected_targets: template.targets.len(),
        actual_targets,
    }
}
