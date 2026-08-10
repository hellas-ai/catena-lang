//! Erase the temporary context projections used while constructing closures.
//!
//! Closure replacement introduces a generated `context.closure.*` operation.
//! It is a bookkeeping boundary, not an executable operation: it separates
//! values captured in the runtime environment from erased values needed to
//! instantiate `name.closure.*`.
//!
//! A representative projection looks like this before erasure:
//!
//! ```text
//! capture ────────────────────────────────▶ capture copy
//! m ───────┐
//! n ───────┴─▶ context.closure.* ─────────▶ n
//!                                           └────────▶ n
//! ```
//!
//! Erasure removes the box and leaves only structural wiring:
//!
//! ```text
//! capture ────────────────────────────────▶ capture copy
//! m ───────▶ discard
//! n ──────────────────────────────────────▶ n
//!   └─────────────────────────────────────▶ n
//! ```
//!
//! Thus the generated operation may preserve values, discard unused context,
//! or reuse one context input more than once. None of those actions has a
//! runtime implementation; the surrounding hypergraph supplies the structural
//! semantics.

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
    #[error("failed to quotient context-free closure conversion `{theory}.{definition}`")]
    Quotient { theory: String, definition: String },
}

#[derive(Clone, Copy, Debug, Default)]
struct EraseContexts;

/// Map a complete closure-converted graph while removing generated context
/// operations locally.
///
/// Objects are unchanged. Ordinary operations are copied verbatim. A generated
/// context operation is replaced by an edge-free open hypergraph describing
/// only how its inputs and outputs are connected. `map_arrow` applies those
/// local replacements throughout a definition; the later quotient joins their
/// boundaries to the surrounding graph.
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

        erase_context_projection(source, target)
    }

    fn map_arrow(&self, term: &OpenHypergraph<Obj, Operation>) -> OpenHypergraph<Obj, Operation> {
        try_define_map_arrow(self, term).expect("validated context erasure should define a functor")
    }
}

/// Replace one generated context operation with structural wiring.
fn erase_context_projection(source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, Operation> {
    // Start with every input exposed as a graph source. Inputs which are not
    // selected below remain sources but have no target, which is weakening
    // (discarding) in the surrounding hypergraph.
    let mut wiring = OpenHypergraph::identity(source.to_vec());

    // Replacement puts runtime captures first and copies them in the same
    // order. Preserve the complete equal prefix positionally. This matters
    // when two captures have the same type: each output must retain its own
    // input wire rather than both choosing the first type-compatible input.
    let unchanged = unchanged_prefix_len(source, target);
    let mut outputs = wiring.sources[..unchanged].to_vec();

    // The remaining outputs are static context for `name.closure.*`. They may
    // contract equal inputs or expose a type-level value derived from context.
    for object in &target[unchanged..] {
        outputs.push(connect_static_context_output(&mut wiring, source, object));
    }

    wiring.targets = outputs;
    wiring
}

fn unchanged_prefix_len(source: &[Obj], target: &[Obj]) -> usize {
    source
        .iter()
        .zip(target)
        .take_while(|(source, target)| source == target)
        .count()
}

/// Connect one static output to an equal input when possible.
///
/// Selecting the same source repeatedly is contraction (copying erased
/// context). If no input has exactly this object, a fresh edge-free node
/// represents a value determined purely at the type level.
fn connect_static_context_output(
    wiring: &mut OpenHypergraph<Obj, Operation>,
    source: &[Obj],
    object: &Obj,
) -> NodeId {
    source
        .iter()
        .position(|candidate| candidate == object)
        .map(|index| wiring.sources[index])
        .unwrap_or_else(|| wiring.new_node(object.clone()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_prefix_is_preserved_positionally() {
        let capture = obj("capture");
        let source = vec![capture.clone(), capture.clone()];

        let erased = erase_context_projection(&source, &source);

        assert!(erased.hypergraph.edges.is_empty());
        assert_eq!(erased.targets, erased.sources);
    }

    #[test]
    fn static_context_can_be_discarded_and_reused() {
        let capture = obj("capture");
        let unused = obj("m");
        let reused = obj("n");
        let source = vec![capture.clone(), unused, reused.clone()];
        let target = vec![capture, reused.clone(), reused];

        let erased = erase_context_projection(&source, &target);

        assert!(erased.hypergraph.edges.is_empty());
        assert_eq!(erased.targets[0], erased.sources[0]);
        assert_eq!(erased.targets[1], erased.sources[2]);
        assert_eq!(erased.targets[2], erased.sources[2]);
        assert!(!erased.targets.contains(&erased.sources[1]));
    }

    #[test]
    fn derived_type_level_output_needs_no_runtime_edge() {
        let source = vec![obj("typed-n")];
        let target = vec![obj("n")];

        let erased = erase_context_projection(&source, &target);

        assert!(erased.hypergraph.edges.is_empty());
        assert_ne!(erased.targets[0], erased.sources[0]);
        assert_eq!(erased.hypergraph.nodes[erased.targets[0].0], target[0]);
    }

    fn obj(name: &str) -> Obj {
        Tree::Node(
            name.parse().expect("test operation should parse"),
            0,
            vec![],
        )
    }
}
