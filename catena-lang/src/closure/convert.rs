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
}

impl ConvertedClosure {
    pub fn name(&self, definition_name: &Operation) -> Operation {
        closure_operation(definition_name, self.node)
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
        let type_info = type_info(&rewritten, &region)?;
        let replacement = replacement_region(
            definition_name,
            &rewritten,
            &region,
            &type_info,
            job.original_wire,
        );
        closures.push(ConvertedClosure {
            node: job.original_wire,
            term: body,
            type_info,
        });

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
    closure_name_wire: NodeId,
) -> AnnotatedTerm {
    let mut replacement = AnnotatedTerm::empty();
    let sources = region
        .leaf_inputs
        .iter()
        .map(|wire| replacement.new_node(definition.hypergraph.nodes[wire.0].clone()))
        .collect::<Vec<_>>();
    let (environment_components, name_sources) = split_sources_for_environment_and_name(
        &mut replacement,
        definition_name,
        closure_name_wire,
        &sources,
    );
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

fn split_sources_for_environment_and_name(
    replacement: &mut AnnotatedTerm,
    definition_name: &Operation,
    closure_name_wire: NodeId,
    sources: &[NodeId],
) -> (Vec<NodeId>, Vec<NodeId>) {
    sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let source_type = replacement.hypergraph.nodes[source.0].clone();
            let environment_component = replacement.new_node(source_type.clone());
            let name_source = replacement.new_node(source_type);
            replacement.new_edge(
                copy_operation(definition_name, closure_name_wire, index),
                (vec![*source], vec![environment_component, name_source]),
            );
            (environment_component, name_source)
        })
        .unzip()
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

fn unit_type() -> Obj {
    Tree::Node(op(UNIT_TYPE), 0, vec![])
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
