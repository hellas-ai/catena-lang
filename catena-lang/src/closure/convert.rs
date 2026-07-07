use std::collections::{BTreeMap, BTreeSet};

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    closure::{
        body::{ClosureBodyError, closure_body},
        extract::{ExtractRegionError, extract_region},
        region::{ClosureRegion, ClosureRegionError, closure_region},
        rewrite::{RewriteRegionError, rewrite_region},
    },
    prefixes::{GENERATED_COPY_PREFIX, NAME_PREFIX},
    stdlib::constants::{
        FN_HOM_TYPE, FN_REF_TYPE, PRODUCT_INTRO, PRODUCT_TYPE, UNIT_INTRO, UNIT_TYPE, VALUE_TYPE,
    },
};

type Obj = Tree<(), Operation>;

#[derive(Debug, Clone)]
pub struct Converted {
    pub definition: AnnotatedTerm,
    pub closures: Vec<ConvertedClosure>,
}

#[derive(Debug, Clone)]
pub struct ConvertedClosure {
    pub node: NodeId,
    pub term: AnnotatedTerm,
    pub type_info: TypeInfo,
    pub context: ClosureContext,
}

impl ConvertedClosure {
    fn new(
        node: NodeId,
        term: AnnotatedTerm,
        type_info: TypeInfo,
        context: ClosureContext,
    ) -> Self {
        context.assert_term_boundary_uses_compact_leaves(&term);
        Self {
            node,
            term,
            type_info,
            context,
        }
    }

    pub fn name(&self, definition_name: &Operation) -> Operation {
        closure_operation(definition_name, self.node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureContext {
    /// Maps compact closure-context leaves back to their original definition leaves.
    ///
    /// Entry `original_leaf_by_compact_leaf[i] = j` means generated closure
    /// body type `Leaf(i)` came from original definition context `Leaf(j)`.
    pub original_leaf_by_compact_leaf: Vec<usize>,
}

impl ClosureContext {
    fn from_term_boundary(term: &AnnotatedTerm) -> Self {
        let mut leaves = BTreeSet::new();
        let boundary_types = interface_types(term, &term.sources)
            .into_iter()
            .chain(interface_types(term, &term.targets));
        for object in boundary_types {
            collect_leaf_indices(object, &mut leaves);
        }
        Self {
            original_leaf_by_compact_leaf: leaves.into_iter().collect(),
        }
    }

    pub fn arity(&self) -> usize {
        self.original_leaf_by_compact_leaf.len()
    }

    fn compact_leaf_by_original_leaf(&self) -> BTreeMap<usize, usize> {
        self.original_leaf_by_compact_leaf
            .iter()
            .copied()
            .enumerate()
            .map(|(local, original)| (original, local))
            .collect()
    }

    fn requires_original_leaf(&self, original_leaf: usize) -> bool {
        self.original_leaf_by_compact_leaf
            .iter()
            .any(|required| *required == original_leaf)
    }

    fn name_sources_from_original_metavars(
        &self,
        metavars_by_original_leaf: &BTreeMap<usize, NodeId>,
    ) -> Vec<NodeId> {
        self.original_leaf_by_compact_leaf
            .iter()
            .map(|original| {
                metavars_by_original_leaf
                    .get(original)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "closure conversion could not connect generated closure name: closure body requires original context leaf {original}, but the closure-region inputs do not contain a top-level Leaf({original})"
                        )
                    })
            })
            .collect()
    }

    fn relabel_term(&self, term: &AnnotatedTerm) -> AnnotatedTerm {
        let compact_leaf_by_original_leaf = self.compact_leaf_by_original_leaf();
        let relabeled = term
            .clone()
            .map_nodes(|object| relabel_object_context(object, &compact_leaf_by_original_leaf));

        self.assert_term_boundary_uses_compact_leaves(&relabeled);

        relabeled
    }

    fn assert_term_boundary_uses_compact_leaves(&self, term: &AnnotatedTerm) {
        assert_eq!(
            ClosureContext::from_term_boundary(term).original_leaf_by_compact_leaf,
            (0..self.arity()).collect::<Vec<_>>(),
            "generated closure body boundary should use exactly Leaf(0)..Leaf(context_arity - 1) after context relabeling"
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub environment: Obj, // X (always packed)
    pub domain: Obj,      // A (always packed)
    pub codomain: Obj,    // B (always packed)
}

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error(transparent)]
    Region(#[from] ClosureRegionError),
    #[error(transparent)]
    Extract(#[from] ExtractRegionError),
    #[error(transparent)]
    Body(#[from] ClosureBodyError),
    #[error(transparent)]
    Rewrite(#[from] RewriteRegionError),
    #[error("pending closure root n{wire} was deleted by an earlier closure rewrite")]
    PendingClosureDeleted { wire: usize },
}

#[derive(Debug, Clone, Copy)]
struct PendingClosure {
    original_wire: NodeId,
    current_wire: NodeId,
}

/// Convert closure-typed output regions of an annotated term.
///
/// Returns the rewritten term plus the generated closure body terms. Each
/// generated closure records the original closure node id and the type
/// information needed by the caller to elaborate its `name.closure.*` operation.
pub fn convert(
    definition_name: &Operation,
    definition: &AnnotatedTerm,
    closure_wires: &[NodeId],
) -> Result<Converted, ConvertError> {
    let mut closures = Vec::new();
    let mut rewritten = definition.clone();
    let mut pending = closure_wires
        .iter()
        .copied()
        .map(|wire| PendingClosure {
            original_wire: wire,
            current_wire: wire,
        })
        .collect::<Vec<_>>();

    while let Some(job) = pending.first().copied() {
        pending.remove(0);

        let [region] = closure_region(&rewritten, &[job.current_wire])?
            .try_into()
            .expect("requested exactly one closure region");
        let extracted = extract_region(&rewritten, &region)?;
        let body = closure_body(&extracted)?;
        let context = ClosureContext::from_term_boundary(&body);
        let body = context.relabel_term(&body);
        let type_info = type_info(&rewritten, &region)?;
        let replacement = replacement_region(
            definition_name,
            &rewritten,
            &region,
            &type_info,
            &context,
            job.original_wire,
        );
        closures.push(ConvertedClosure::new(
            job.original_wire,
            body,
            type_info,
            context,
        ));

        let rewrite = rewrite_region(&rewritten, &region, &replacement)?;
        pending = remap_pending_closures(pending, &rewrite.node_map)?;
        rewritten = rewrite.definition;
    }

    Ok(Converted {
        definition: rewritten,
        closures,
    })
}

fn remap_pending_closures(
    pending: Vec<PendingClosure>,
    node_map: &[Option<usize>],
) -> Result<Vec<PendingClosure>, ConvertError> {
    pending
        .into_iter()
        .map(|job| {
            let current_wire = node_map
                .get(job.current_wire.0)
                .and_then(|mapped| mapped.map(NodeId))
                .ok_or(ConvertError::PendingClosureDeleted {
                    wire: job.current_wire.0,
                })?;
            Ok(PendingClosure {
                original_wire: job.original_wire,
                current_wire,
            })
        })
        .collect()
}

fn replacement_region(
    definition_name: &Operation,
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    type_info: &TypeInfo,
    context: &ClosureContext,
    closure_name_wire: NodeId,
) -> AnnotatedTerm {
    let mut replacement = AnnotatedTerm::empty();

    // Expose the region leaf inputs as replacement sources.
    let sources = region
        .leaf_inputs
        .iter()
        .map(|wire| replacement.new_node(definition.hypergraph.nodes[wire.0].clone()))
        .collect::<Vec<_>>();

    // Build environment components and collect generated-name metavars.
    // Every leaf input contributes to the runtime environment. Only top-level metavar leaves required by the closure context also feed `name.closure.*`.
    //
    //   x: val ----------------------------> environment component
    //
    //   n: Leaf(2) -- copy.closure.*.i ----> environment component
    //              \
    //               `----------------------> name metavar for original Leaf(2)
    let mut environment_components = Vec::new();
    let mut name_metavars_by_original_leaf = BTreeMap::new();
    for (index, source) in sources.iter().copied().enumerate() {
        if let Some(original_leaf) = top_level_metavar_leaf(&replacement, source)
            && context.requires_original_leaf(original_leaf)
        {
            let (environment_component, name_metavar) = copy_metavar_for_environment_and_name(
                &mut replacement,
                definition_name,
                closure_name_wire,
                index,
                source,
            );
            environment_components.push(environment_component);
            insert_name_metavar(
                &mut name_metavars_by_original_leaf,
                original_leaf,
                name_metavar,
            );
            continue;
        }

        environment_components.push(source);
    }

    // Order the collected name metavars using the closure context.
    //
    //   context.original_leaf_by_compact_leaf = [2, 5]
    //
    // means:
    //
    //   compact Leaf(0) expects original Leaf(2)
    //   compact Leaf(1) expects original Leaf(5)
    //
    // so the `name.closure.*` source list is ordered as:
    //
    //   [metavar for original Leaf(2), metavar for original Leaf(5)]
    let name_sources = context.name_sources_from_original_metavars(&name_metavars_by_original_leaf);
    assert_eq!(
        name_sources.len(),
        context.arity(),
        "generated closure name inputs should match compact closure context arity"
    );

    // Pack the environment, create the function-pointer node, and connect the
    // generated closure name.
    //
    //   environment components ----> packed environment
    //
    //   ordered name metavars -----> name.closure.* ----> function pointer
    //
    // replacement targets are:
    //
    //   [packed environment, function pointer]
    let environment = packed_environment_target(
        &mut replacement,
        &environment_components,
        &type_info.environment,
    );
    let function_pointer = replacement.new_node(function_pointer_type(
        vec![type_info.environment.clone(), type_info.domain.clone()],
        vec![type_info.codomain.clone()],
    ));
    replacement.new_edge(
        name_operation(definition_name, closure_name_wire),
        (name_sources, vec![function_pointer]),
    );
    replacement.sources = sources;
    replacement.targets = vec![environment, function_pointer];
    replacement
}

fn insert_name_metavar(
    name_metavars_by_original_leaf: &mut BTreeMap<usize, NodeId>,
    original_leaf: usize,
    name_metavar: NodeId,
) {
    // A closure region can mention the same top-level context leaf through multiple boundary
    // wires. In `matmul-f32-inner`, for example, `b-for-product`, `b-for-unit`, and
    // `b-for-identity` are separate region inputs but all refer to the same top-level `b` leaf.
    // The generated `name.*` operation only has one metavariable for that leaf, so keep the first
    // representative here. The duplicated runtime/type wires still remain in the closure
    // environment; this only deduplicates the name operation's context interface.
    name_metavars_by_original_leaf
        .entry(original_leaf)
        .or_insert(name_metavar);
}

fn copy_metavar_for_environment_and_name(
    replacement: &mut AnnotatedTerm,
    definition_name: &Operation,
    closure_name_wire: NodeId,
    index: usize,
    source: NodeId,
) -> (NodeId, NodeId) {
    let source_type = replacement.hypergraph.nodes[source.0].clone();
    let environment_component = replacement.new_node(source_type.clone());
    let name_metavar = replacement.new_node(source_type);
    replacement.new_edge(
        copy_operation(definition_name, closure_name_wire, index),
        (vec![source], vec![environment_component, name_metavar]),
    );
    (environment_component, name_metavar)
}

fn top_level_metavar_leaf(replacement: &AnnotatedTerm, source: NodeId) -> Option<usize> {
    match &replacement.hypergraph.nodes[source.0] {
        Tree::Leaf(original, _) => Some(*original),
        _ => None,
    }
}

fn packed_environment_target(
    replacement: &mut AnnotatedTerm,
    components: &[NodeId],
    environment_type: &Obj,
) -> NodeId {
    match components {
        [] => {
            let unit = replacement.new_node(unit_type());
            replacement.new_edge(op(UNIT_INTRO), (vec![], vec![unit]));
            unit
        }
        [only] => *only,
        _ => {
            let component_types = components
                .iter()
                .map(|node| replacement.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>();
            let packed = replacement.new_node(environment_type.clone());
            pack_environment(replacement, components, &component_types, packed);
            packed
        }
    }
}

fn pack_environment(
    replacement: &mut AnnotatedTerm,
    components: &[NodeId],
    component_types: &[Obj],
    packed: NodeId,
) {
    match components {
        [] | [_] => {}
        [left, right] => {
            replacement.new_edge(op(PRODUCT_INTRO), (vec![*left, *right], vec![packed]));
        }
        [left, rest @ ..] => {
            let tail_type = pack_object(component_types[1..].to_vec());
            let tail = replacement.new_node(tail_type);
            pack_environment(replacement, rest, &component_types[1..], tail);
            replacement.new_edge(op(PRODUCT_INTRO), (vec![*left, tail], vec![packed]));
        }
    }
}

fn type_info(definition: &AnnotatedTerm, region: &ClosureRegion) -> Result<TypeInfo, ConvertError> {
    let environment = pack_object(
        region
            .leaf_inputs
            .iter()
            .map(|wire| definition.hypergraph.nodes[wire.0].clone())
            .collect(),
    );
    let (domain, codomain) = closure_parts(&region.closure_type)
        .expect("closure region type should be a binary closure type");
    Ok(TypeInfo {
        environment,
        domain: domain.clone(),
        codomain: codomain.clone(),
    })
}

fn relabel_object_context(
    object: Obj,
    compact_leaf_by_original_leaf: &BTreeMap<usize, usize>,
) -> Obj {
    match object {
        Tree::Empty => Tree::Empty,
        Tree::Leaf(original, annotation) => {
            let local = compact_leaf_by_original_leaf
                .get(&original)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "closure conversion cannot relabel context leaf {original}: it is not part of the closure boundary context"
                    )
                });
            Tree::Leaf(local, annotation)
        }
        Tree::Node(operation, annotation, children) => Tree::Node(
            operation,
            annotation,
            children
                .into_iter()
                .map(|child| relabel_object_context(child, compact_leaf_by_original_leaf))
                .collect(),
        ),
    }
}

fn closure_operation(definition_name: &Operation, closure_wire: NodeId) -> Operation {
    format!("closure.{}.{}", definition_name, closure_wire.0)
        .parse()
        .expect("generated closure operation should parse")
}

fn name_operation(definition_name: &Operation, closure_wire: NodeId) -> Operation {
    format!(
        "{NAME_PREFIX}{}",
        closure_operation(definition_name, closure_wire)
    )
    .parse()
    .expect("generated name operation should parse")
}

fn copy_operation(definition_name: &Operation, closure_wire: NodeId, index: usize) -> Operation {
    format!(
        "{GENERATED_COPY_PREFIX}closure.{}.{}.{}",
        definition_name, closure_wire.0, index
    )
    .parse()
    .expect("generated copy operation should parse")
}

fn closure_parts(object: &Obj) -> Option<(&Obj, &Obj)> {
    let Tree::Node(operation, _, children) = object else {
        return None;
    };
    if operation.as_str() != FN_HOM_TYPE {
        return None;
    }
    let [domain, codomain] = children.as_slice() else {
        return None;
    };
    Some((domain, codomain))
}

fn function_pointer_type(sources: Vec<Obj>, targets: Vec<Obj>) -> Obj {
    value_type(function_type(pack_object(sources), pack_object(targets)))
}

fn function_type(domain: Obj, codomain: Obj) -> Obj {
    Tree::Node(op(FN_REF_TYPE), 0, vec![domain, codomain])
}

fn value_type(inner: Obj) -> Obj {
    Tree::Node(op(VALUE_TYPE), 0, vec![inner])
}

fn pack_object(objects: Vec<Obj>) -> Obj {
    match objects.as_slice() {
        [] => Tree::Node(op(UNIT_TYPE), 0, vec![]),
        [only] => only.clone(),
        [head, tail @ ..] => Tree::Node(
            op(PRODUCT_TYPE),
            0,
            vec![head.clone(), pack_object(tail.to_vec())],
        ),
    }
}

fn collect_leaf_indices(object: &Obj, indices: &mut BTreeSet<usize>) {
    match object {
        Tree::Empty => {}
        Tree::Leaf(index, _) => {
            indices.insert(*index);
        }
        Tree::Node(_, _, children) => {
            for child in children {
                collect_leaf_indices(child, indices);
            }
        }
    }
}

fn unit_type() -> Obj {
    Tree::Node(op(UNIT_TYPE), 0, vec![])
}

fn interface_types<'a>(term: &'a AnnotatedTerm, interface: &[NodeId]) -> Vec<&'a Obj> {
    interface
        .iter()
        .map(|node| &term.hypergraph.nodes[node.0])
        .collect()
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closure_context_collects_sorted_unique_boundary_leaves() {
        let mut term = AnnotatedTerm::empty();
        let source_a = term.new_node(Tree::Leaf(5, ()));
        let source_b = term.new_node(obj("val", vec![obj("ix", vec![Tree::Leaf(2, ())])]));
        let target = term.new_node(obj(
            FN_HOM_TYPE,
            vec![
                obj("val", vec![obj("ix", vec![Tree::Leaf(5, ())])]),
                obj("val", vec![obj("u64", vec![])]),
            ],
        ));
        term.sources = vec![source_a, source_b];
        term.targets = vec![target];

        let context = ClosureContext::from_term_boundary(&term);

        assert_eq!(
            context.original_leaf_by_compact_leaf,
            vec![2, 5],
            "closure context should use sorted unique original leaves from source and target boundary types"
        );
        assert_eq!(context.arity(), 2);
    }

    #[test]
    fn closure_context_is_nullary_when_boundary_has_no_leaves() {
        let mut term = AnnotatedTerm::empty();
        let source = term.new_node(obj("1", vec![]));
        let target = term.new_node(obj("val", vec![obj("u64", vec![])]));
        term.sources = vec![source];
        term.targets = vec![target];

        let context = ClosureContext::from_term_boundary(&term);

        assert_eq!(context.original_leaf_by_compact_leaf, Vec::<usize>::new());
        assert_eq!(context.arity(), 0);
        assert!(context.compact_leaf_by_original_leaf().is_empty());
    }

    #[test]
    fn closure_context_builds_reverse_compact_leaf_lookup() {
        let context = ClosureContext {
            original_leaf_by_compact_leaf: vec![2, 5],
        };

        assert_eq!(
            context.compact_leaf_by_original_leaf(),
            BTreeMap::from([(2, 0), (5, 1)]),
            "reverse lookup should map original leaves to compact generated closure leaves"
        );
    }

    #[test]
    fn closure_context_relabels_term_boundary_to_compact_leaves() {
        let mut term = AnnotatedTerm::empty();
        let source = term.new_node(Tree::Leaf(5, ()));
        let target = term.new_node(obj(
            FN_HOM_TYPE,
            vec![
                obj("val", vec![obj("ix", vec![Tree::Leaf(2, ())])]),
                obj("val", vec![obj("ix", vec![Tree::Leaf(5, ())])]),
            ],
        ));
        term.sources = vec![source];
        term.targets = vec![target];

        let context = ClosureContext::from_term_boundary(&term);
        let relabeled = context.relabel_term(&term);

        assert_eq!(context.original_leaf_by_compact_leaf, vec![2, 5]);
        assert_eq!(
            interface_types(&relabeled, &relabeled.sources),
            vec![&Tree::Leaf(1, ())],
            "original Leaf(5) should become compact Leaf(1)"
        );
        assert_eq!(
            interface_types(&relabeled, &relabeled.targets),
            vec![&obj(
                FN_HOM_TYPE,
                vec![
                    obj("val", vec![obj("ix", vec![Tree::Leaf(0, ())])]),
                    obj("val", vec![obj("ix", vec![Tree::Leaf(1, ())])]),
                ],
            )],
            "boundary leaves should be relabeled according to the compact context"
        );
        context.assert_term_boundary_uses_compact_leaves(&relabeled);
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }
}
