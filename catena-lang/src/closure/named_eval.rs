//! Specialize named evaluations before closure-region conversion.
//!
//! `forget_closures` preserves a statically known lifted call as one
//! [`ClosureForgotten::NamedEval`] edge whose boundary already matches the
//! forgotten boundary of the callee. The procedure is deliberately independent
//! of closure regions:
//!
//! 1. find a `NamedEval` edge;
//! 2. instantiate the named definition's forgotten template;
//! 3. verify the flattened call boundary;
//! 4. splice the template in place of the edge.

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryArrow, TheoryId, TheorySet},
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
        // Always inline from the original map. A template may itself contain a
        // NamedEval, and mutating it while processing another caller would make
        // specialization depend on map iteration order.
        let theory_templates = &forgotten[theory_id];

        for (caller, term) in definitions.iter_mut() {
            specialize_definition(theory_id, caller, term, theory_templates)?;
        }

        remove_closure_boundary_templates(definitions, arrows);
    }

    Ok(output)
}

#[derive(Debug, Clone)]
struct NamedEvalSite {
    edge: EdgeId,
    definition: Operation,
    context_arity: usize,
}

fn specialize_definition(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    templates: &BTreeMap<Operation, ClosureForgottenTerm>,
) -> Result<(), NamedEvalError> {
    // Splicing may reveal NamedEval edges from the inserted template, so keep
    // going until the definition contains only ordinary operations/markers.
    while let Some(site) = find_next(term) {
        let template =
            templates
                .get(&site.definition)
                .ok_or_else(|| NamedEvalError::MissingBody {
                    theory: theory_id.to_string(),
                    definition: site.definition.to_string(),
                })?;
        replace_at_site(theory_id, caller, term, template, site)?;
    }
    Ok(())
}

fn find_next(term: &ClosureForgottenTerm) -> Option<NamedEvalSite> {
    term.hypergraph
        .edges
        .iter()
        .enumerate()
        .find_map(|(edge, operation)| match operation {
            ClosureForgotten::NamedEval {
                definition,
                context_arity,
            } => Some(NamedEvalSite {
                edge: EdgeId(edge),
                definition: definition.clone(),
                context_arity: *context_arity,
            }),
            ClosureForgotten::Operation(_) | ClosureForgotten::ClosureMarker => None,
        })
}

fn replace_at_site(
    theory_id: &TheoryId,
    caller: &Operation,
    term: &mut ClosureForgottenTerm,
    template: &ClosureForgottenTerm,
    site: NamedEvalSite,
) -> Result<(), NamedEvalError> {
    let boundary = term.hypergraph.adjacency[site.edge.0].clone();
    if boundary.sources.len() < site.context_arity {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &site.definition,
            template,
            0,
            boundary.targets.len(),
        ));
    }
    let context_types = node_types(term, &boundary.sources[..site.context_arity]);
    let call_inputs = boundary.sources[site.context_arity..].to_vec();
    let call_outputs = boundary.targets;
    let template = instantiate_template(template, &context_types);

    if !boundary_matches(term, &call_inputs, &call_outputs, &template) {
        return Err(boundary_mismatch(
            theory_id,
            caller,
            &site.definition,
            &template,
            call_inputs.len(),
            call_outputs.len(),
        ));
    }

    splice_template(term, site.edge, call_inputs, call_outputs, template);
    Ok(())
}

fn node_types(term: &ClosureForgottenTerm, nodes: &[NodeId]) -> Vec<Tree<(), Operation>> {
    nodes
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn instantiate_template(
    template: &ClosureForgottenTerm,
    context_types: &[Tree<(), Operation>],
) -> ClosureForgottenTerm {
    template
        .clone()
        .map_nodes(|object| instantiate_object(&object, context_types))
}

fn boundary_matches(
    caller: &ClosureForgottenTerm,
    call_inputs: &[NodeId],
    call_outputs: &[NodeId],
    template: &ClosureForgottenTerm,
) -> bool {
    call_inputs.len() == template.sources.len()
        && call_outputs.len() == template.targets.len()
        && same_types(caller, call_inputs, template, &template.sources)
        && same_types(caller, call_outputs, template, &template.targets)
}

fn splice_template(
    term: &mut ClosureForgottenTerm,
    edge: EdgeId,
    call_inputs: Vec<NodeId>,
    call_outputs: Vec<NodeId>,
    template: ClosureForgottenTerm,
) {
    term.delete_edges(&[edge]);
    let (template_sources, template_targets) = term.append(template);
    for (call_input, template_input) in call_inputs.into_iter().zip(template_sources) {
        term.unify(call_input, template_input);
    }
    for (template_output, call_output) in template_targets.into_iter().zip(call_outputs) {
        term.unify(template_output, call_output);
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

fn remove_closure_boundary_templates(
    definitions: &mut BTreeMap<Operation, ClosureForgottenTerm>,
    arrows: &BTreeMap<Operation, TheoryArrow>,
) {
    // These definitions exist only as specialization templates. Their original
    // closure-bearing ABI is not a runtime/codegen entry point.
    definitions.retain(|definition, _| {
        arrows
            .get(definition)
            .is_none_or(|arrow| !arrow_has_closure_boundary(arrow))
    });
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

#[cfg(test)]
mod tests {
    use metacat::theory::{RawTheorySet, TheorySet};

    use crate::pass::forget_closures::ClosureForgotten;

    #[test]
    fn substitutes_named_eval_and_removes_its_closure_boundary_template() {
        let source = r#"
            (def program apply-closure : ({1 (bool val)} =>) -> (bool val)
              = run)
            (def program use-named-closure : (bool val) -> (bool val)
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

        assert!(
            forgotten[&program][&use_named]
                .hypergraph
                .edges
                .iter()
                .any(|edge| matches!(
                    edge,
                    ClosureForgotten::NamedEval { definition, .. }
                        if definition == &apply_closure
                ))
        );

        let specialized = super::run(&theory_set, &forgotten)
            .expect("named closure-boundary evaluation should specialize");
        assert!(
            specialized[&program][&use_named]
                .hypergraph
                .edges
                .iter()
                .all(|edge| !matches!(edge, ClosureForgotten::NamedEval { .. }))
        );
        assert!(!specialized[&program].contains_key(&apply_closure));
        assert!(specialized[&program].contains_key(&use_named));
    }
}
