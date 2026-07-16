//! Erase generated context projections from the runtime closure representation.

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{theory::TheoryId, tree::Tree};
use open_hypergraphs::lax::{
    NodeId, OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};
use thiserror::Error;

use crate::check::AnnotatedTerm;
use crate::{prefixes::GENERATED_CONTEXT_PREFIX, report::TheoryTermMap};

type Obj = Tree<(), Operation>;

#[derive(Debug, Error)]
pub enum EraseContextsError {
    #[error("generated closure context `{operation}` does not preserve its environment inputs")]
    InvalidBoundary { operation: String },
    #[error("failed to quotient context-free closure conversion `{theory}.{definition}`")]
    Quotient { theory: String, definition: String },
}

#[derive(Clone, Copy, Debug, Default)]
struct EraseContexts;

impl Functor<Obj, Operation, Obj, Operation> for EraseContexts {
    fn map_object(&self, object: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(object.clone())
    }

    fn map_operation(
        &self,
        operation: &Operation,
        source: &[Obj],
        target: &[Obj],
    ) -> OpenHypergraph<Obj, Operation> {
        if !operation.as_str().starts_with(GENERATED_CONTEXT_PREFIX) {
            return OpenHypergraph::singleton(operation.clone(), source.to_vec(), target.to_vec());
        }

        assert!(
            target.starts_with(source),
            "validated generated context should preserve its environment inputs"
        );
        let mut result = OpenHypergraph::identity(source.to_vec());
        let mut extra_targets = Vec::new();
        for object in &target[source.len()..] {
            let node = source
                .iter()
                .position(|source| source == object)
                .map(NodeId)
                .unwrap_or_else(|| result.new_node(object.clone()));
            extra_targets.push(node);
        }
        result.targets.extend(extra_targets);
        result
    }

    fn map_arrow(&self, term: &OpenHypergraph<Obj, Operation>) -> OpenHypergraph<Obj, Operation> {
        try_define_map_arrow(self, term).expect("validated context erasure should define a functor")
    }
}

pub fn erase(terms: &TheoryTermMap) -> Result<TheoryTermMap, EraseContextsError> {
    terms
        .iter()
        .map(|(theory_id, definitions)| erase_theory(theory_id, definitions))
        .collect()
}

fn erase_theory(
    theory_id: &TheoryId,
    definitions: &BTreeMap<Operation, AnnotatedTerm>,
) -> Result<(TheoryId, BTreeMap<Operation, AnnotatedTerm>), EraseContextsError> {
    let definitions = definitions
        .iter()
        .map(|(definition_name, term)| {
            validate_context_boundaries(term)?;
            let mut transformed = EraseContexts.map_arrow(term);
            transformed
                .quotient()
                .map_err(|_| EraseContextsError::Quotient {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                })?;
            Ok((definition_name.clone(), transformed))
        })
        .collect::<Result<_, EraseContextsError>>()?;
    Ok((theory_id.clone(), definitions))
}

fn validate_context_boundaries(term: &AnnotatedTerm) -> Result<(), EraseContextsError> {
    for (operation, boundary) in term
        .hypergraph
        .edges
        .iter()
        .zip(&term.hypergraph.adjacency)
        .filter(|(operation, _)| operation.as_str().starts_with(GENERATED_CONTEXT_PREFIX))
    {
        let source = boundary
            .sources
            .iter()
            .map(|node| &term.hypergraph.nodes[node.0]);
        let target = boundary
            .targets
            .iter()
            .take(boundary.sources.len())
            .map(|node| &term.hypergraph.nodes[node.0]);
        if boundary.targets.len() < boundary.sources.len() || !source.eq(target) {
            return Err(EraseContextsError::InvalidBoundary {
                operation: operation.to_string(),
            });
        }
    }
    Ok(())
}
