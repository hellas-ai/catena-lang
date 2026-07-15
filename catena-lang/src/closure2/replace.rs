//! Replace discovered closure regions with explicit environments and function pointers.

use std::collections::{BTreeMap, BTreeSet};

use hexpr::{Hexpr, Operation, Variable, interpret::Error as HexprInterpretError, try_interpret};
use metacat::{
    check::eval_type,
    dual::Dual,
    spiders::WithSpiders,
    theory::{Term, Theory, TheoryArrow, TheorySet, ast::RawTheoryArrow, model::SignatureError},
    tree::Tree,
};
use open_hypergraphs::lax::{EdgeId, Hyperedge, NodeId};
use thiserror::Error;

use crate::{
    check::TypedTerm,
    closure2::{
        definition::closure_operation,
        region::{ClosureRegion, ClosureRegionMap, find_regions},
    },
    hexpr::{objects_to_hexpr, term_to_hexpr},
    nonstrict::to_packer,
    pass::forget_closures::{Region, RegionTerm},
    prefixes::{GENERATED_CONTEXT_PREFIX, GENERATED_VARIABLE_PREFIX, NAME_PREFIX},
    report::TheoryTermMap,
};

type Obj = Tree<(), Operation>;

const CONVERTED_PRIMITIVES: &[(&str, &str)] = &[
    ("if", "ifc"),
    ("bool.if", "bool.ifc"),
    ("reduce", "reducec"),
];

#[derive(Debug, Clone)]
pub struct Replacement {
    pub theory_set: TheorySet,
    pub terms: TheoryTermMap,
}

#[derive(Debug, Error)]
pub enum ReplaceClosuresError {
    #[error("missing theory `{0}`")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing syntax theory `{0}`")]
    MissingSyntaxTheory(String),
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("missing generated name operation `{operation}`")]
    MissingNameOperation { operation: String },
    #[error("generated name operation `{operation}` has {targets} targets; expected one")]
    InvalidNameTargets { operation: String, targets: usize },
    #[error("closure region count changed while rewriting `{theory}.{definition}`")]
    RegionCountChanged { theory: String, definition: String },
    #[error("region node w{node} is out of bounds")]
    NodeOutOfBounds { node: usize },
    #[error("region edge e{edge} is out of bounds")]
    EdgeOutOfBounds { edge: usize },
    #[error("a retained boundary references deleted node w{node}")]
    DeletedBoundaryNode { node: usize },
    #[error("generated name context Leaf({leaf}) has no corresponding original context leaf")]
    MissingOriginalContextLeaf { leaf: usize },
    #[error("closure marker remains after replacement")]
    RemainingClosureMarker,
    #[error("failed to quotient replacement for `{theory}.{definition}`: {error}")]
    Quotient {
        theory: String,
        definition: String,
        error: String,
    },
    #[error("failed to interpret generated type map `{map}`: {error}")]
    TypeMapInterpretation {
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
    #[error("generated type maps have incompatible context domains")]
    TypeMapDomainMismatch,
    #[error("could not evaluate generated name type map: {0}")]
    TypeMapEvaluation(String),
}

/// Replace every `!closure` marker and its body in the forgotten definitions.
///
/// The generated `name.closure.*` declaration is the source of truth for the
/// static context inputs and function-pointer output used by each replacement.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<Region<Operation>>,
    regions: &ClosureRegionMap,
) -> Result<Replacement, ReplaceClosuresError> {
    let mut output = theory_set.clone();
    let mut terms = BTreeMap::new();

    for (theory_id, definitions) in forgotten {
        let theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| ReplaceClosuresError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { syntax, arrows } = theory else {
            return Err(ReplaceClosuresError::NotUserTheory(theory_id.to_string()));
        };
        let syntax_theory = theory_set
            .theories
            .get(syntax)
            .ok_or_else(|| ReplaceClosuresError::MissingSyntaxTheory(syntax.to_string()))?;
        let discovered = regions
            .get(theory_id)
            .ok_or_else(|| ReplaceClosuresError::MissingTheory(theory_id.to_string()))?;
        let mut replaced_definitions = BTreeMap::new();

        for (definition_name, term) in definitions {
            let original_arrow = arrows.get(definition_name).ok_or_else(|| {
                ReplaceClosuresError::MissingDefinition {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                }
            })?;
            let definition_regions = discovered
                .get(definition_name)
                .map(Vec::as_slice)
                .unwrap_or_default();
            if definition_regions.is_empty() {
                replaced_definitions
                    .insert(definition_name.clone(), unwrap_operations(term.clone())?);
                continue;
            }
            let mut rewritten = term.clone();

            for original_region in definition_regions {
                let current_regions = find_regions(&rewritten).map_err(|_| {
                    ReplaceClosuresError::RegionCountChanged {
                        theory: theory_id.to_string(),
                        definition: definition_name.to_string(),
                    }
                })?;
                let Some(current_region) = current_regions.first() else {
                    return Err(ReplaceClosuresError::RegionCountChanged {
                        theory: theory_id.to_string(),
                        definition: definition_name.to_string(),
                    });
                };
                let replacement = replacement_term(
                    definition_name,
                    arrows,
                    &rewritten,
                    current_region,
                    original_region,
                )?;
                rewritten = rewrite_one(&rewritten, current_region, &replacement)?;
            }

            if !find_regions(&rewritten)
                .map_err(|_| ReplaceClosuresError::RemainingClosureMarker)?
                .is_empty()
            {
                return Err(ReplaceClosuresError::RegionCountChanged {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                });
            }

            let mut rewritten = unwrap_operations(rewritten)?;
            rewrite_converted_primitives(&mut rewritten);
            rewritten
                .quotient()
                .map_err(|error| ReplaceClosuresError::Quotient {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                    error: format!("{error:?}"),
                })?;
            let Theory::Theory { arrows, .. } = output
                .theories
                .get_mut(theory_id)
                .expect("validated theory should remain present")
            else {
                unreachable!("validated user theory should remain a user theory")
            };
            declare_context_arrows(
                syntax_theory,
                arrows,
                &rewritten,
                original_arrow.type_maps.0.sources.len(),
            )?;
            let arrow = arrows
                .get_mut(definition_name)
                .expect("validated definition should remain present");
            arrow.raw.definition = Some(term_to_hexpr(&rewritten));
            arrow.definition = Some(rewritten.clone().map_nodes(|_| ()));
            replaced_definitions.insert(definition_name.clone(), rewritten);
        }
        terms.insert(theory_id.clone(), replaced_definitions);
    }

    Ok(Replacement {
        theory_set: output,
        terms,
    })
}

fn replacement_term(
    definition_name: &Operation,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    term: &RegionTerm,
    region: &ClosureRegion,
    original_region: &ClosureRegion,
) -> Result<RegionTerm, ReplaceClosuresError> {
    let name_operation = name_operation(definition_name, original_region.closure);
    let name_arrow =
        arrows
            .get(&name_operation)
            .ok_or_else(|| ReplaceClosuresError::MissingNameOperation {
                operation: name_operation.to_string(),
            })?;
    let name_source_types = interface_types(&name_arrow.type_maps.0)?;
    let name_target_types = interface_types(&name_arrow.type_maps.1)?;
    let [name_target_type] = name_target_types.as_slice() else {
        return Err(ReplaceClosuresError::InvalidNameTargets {
            operation: name_operation.to_string(),
            targets: name_target_types.len(),
        });
    };

    let mut original_leaves = BTreeSet::new();
    for node in region
        .environment
        .iter()
        .chain([&region.domain, &region.codomain])
    {
        let object = term
            .hypergraph
            .nodes
            .get(node.0)
            .ok_or(ReplaceClosuresError::NodeOutOfBounds { node: node.0 })?;
        collect_leaf_indices(object, &mut original_leaves);
    }
    let original_leaves = original_leaves.into_iter().collect::<Vec<_>>();

    let mut replacement = RegionTerm::empty();
    let sources = region
        .environment
        .iter()
        .map(|node| {
            term.hypergraph
                .nodes
                .get(node.0)
                .cloned()
                .map(|object| replacement.new_node(object))
                .ok_or(ReplaceClosuresError::NodeOutOfBounds { node: node.0 })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let environment_components = sources
        .iter()
        .map(|source| replacement.new_node(replacement.hypergraph.nodes[source.0].clone()))
        .collect::<Vec<_>>();
    let name_sources = name_source_types
        .iter()
        .map(|object| {
            instantiate_context(object, &original_leaves).map(|object| replacement.new_node(object))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut context_targets = environment_components.clone();
    context_targets.extend(name_sources.iter().copied());
    replacement.new_edge(
        Region::Operation(context_operation(definition_name, original_region.closure)),
        (sources.clone(), context_targets),
    );

    let environment_types = environment_components
        .iter()
        .map(|node| replacement.hypergraph.nodes[node.0].clone())
        .collect();
    let packer = to_packer(environment_types).map_edges(Region::Operation);
    let (packer_sources, packer_targets) = replacement.append(packer);
    for (component, packer_source) in environment_components.into_iter().zip(packer_sources) {
        replacement.unify(component, packer_source);
    }
    let [environment] = packer_targets.as_slice() else {
        unreachable!("environment packer should have one target")
    };

    let function_pointer =
        replacement.new_node(instantiate_context(name_target_type, &original_leaves)?);
    replacement.new_edge(
        Region::Operation(name_operation),
        (name_sources, vec![function_pointer]),
    );
    replacement.sources = sources;
    replacement.targets = vec![*environment, function_pointer];
    Ok(replacement)
}

fn rewrite_one(
    definition: &RegionTerm,
    region: &ClosureRegion,
    replacement: &RegionTerm,
) -> Result<RegionTerm, ReplaceClosuresError> {
    for edge in region.edges.iter().chain([&region.marker]) {
        if edge.0 >= definition.hypergraph.edges.len() {
            return Err(ReplaceClosuresError::EdgeOutOfBounds { edge: edge.0 });
        }
    }
    if region.closure.0 >= definition.hypergraph.nodes.len() {
        return Err(ReplaceClosuresError::NodeOutOfBounds {
            node: region.closure.0,
        });
    }

    let environment = region
        .environment
        .iter()
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();
    let mut deleted_nodes = region
        .nodes
        .iter()
        .copied()
        .filter(|node| !environment.contains(&node.0))
        .collect::<Vec<_>>();
    deleted_nodes.push(region.closure);
    deleted_nodes.sort_by_key(|node| node.0);
    deleted_nodes.dedup();
    let deleted_node_set = deleted_nodes
        .iter()
        .filter(|node| **node != region.closure)
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();

    let mut deleted_edges = region.edges.clone();
    deleted_edges.push(region.marker);
    for (index, boundary) in definition.hypergraph.adjacency.iter().enumerate() {
        if matches!(definition.hypergraph.edges[index], Region::Closure) {
            continue;
        }
        if boundary
            .sources
            .iter()
            .chain(&boundary.targets)
            .any(|node| deleted_node_set.contains(&node.0))
        {
            deleted_edges.push(EdgeId(index));
        }
    }
    deleted_edges.sort_by_key(|edge| edge.0);
    deleted_edges.dedup();
    let deleted_edge_set = deleted_edges
        .iter()
        .map(|edge| edge.0)
        .collect::<BTreeSet<_>>();
    let retained_edges = (0..definition.hypergraph.edges.len())
        .filter(|edge| !deleted_edge_set.contains(edge))
        .map(EdgeId)
        .collect::<Vec<_>>();

    let mut rewritten = definition.clone();
    rewritten.delete_edges(&deleted_edges);
    let node_map = rewritten.hypergraph.delete_nodes_witness(&deleted_nodes);
    rewritten.sources = remap_nodes(&node_map, &definition.sources, region.closure, &[])?;

    let replacement_sources = region
        .environment
        .iter()
        .map(|node| remap_node(&node_map, *node))
        .collect::<Result<Vec<_>, _>>()?;
    let (appended_sources, appended_targets) = rewritten.append(replacement.clone());
    for (outer, inner) in replacement_sources.into_iter().zip(appended_sources) {
        rewritten.unify(outer, inner);
    }

    for (new_edge, old_edge) in retained_edges.iter().enumerate() {
        let old = &definition.hypergraph.adjacency[old_edge.0];
        rewritten.hypergraph.adjacency[new_edge] = Hyperedge {
            sources: remap_nodes(&node_map, &old.sources, region.closure, &appended_targets)?,
            targets: remap_nodes(&node_map, &old.targets, region.closure, &appended_targets)?,
        };
    }
    rewritten.targets = remap_nodes(
        &node_map,
        &definition.targets,
        region.closure,
        &appended_targets,
    )?;
    Ok(rewritten)
}

fn remap_nodes(
    node_map: &[Option<usize>],
    nodes: &[NodeId],
    replaced: NodeId,
    replacement: &[NodeId],
) -> Result<Vec<NodeId>, ReplaceClosuresError> {
    let mut output = Vec::new();
    for &node in nodes {
        if node == replaced {
            output.extend_from_slice(replacement);
        } else {
            output.push(remap_node(node_map, node)?);
        }
    }
    Ok(output)
}

fn remap_node(node_map: &[Option<usize>], node: NodeId) -> Result<NodeId, ReplaceClosuresError> {
    node_map
        .get(node.0)
        .and_then(|node| node.map(NodeId))
        .ok_or(ReplaceClosuresError::DeletedBoundaryNode { node: node.0 })
}

fn unwrap_operations(term: RegionTerm) -> Result<TypedTerm, ReplaceClosuresError> {
    if term
        .hypergraph
        .edges
        .iter()
        .any(|edge| matches!(edge, Region::Closure))
    {
        return Err(ReplaceClosuresError::RemainingClosureMarker);
    }
    Ok(term.map_edges(|edge| match edge {
        Region::Operation(operation) => operation,
        Region::Closure => unreachable!("checked above"),
    }))
}

fn rewrite_converted_primitives(term: &mut TypedTerm) {
    for operation in &mut term.hypergraph.edges {
        if let Some((_, converted)) = CONVERTED_PRIMITIVES
            .iter()
            .find(|(source, _)| operation.as_str() == *source)
        {
            *operation = converted.parse().expect("converted primitive should parse");
        }
    }
}

fn declare_context_arrows(
    syntax: &Theory,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition: &TypedTerm,
    ambient_context_arity: usize,
) -> Result<(), ReplaceClosuresError> {
    for (operation, boundary) in definition
        .hypergraph
        .edges
        .iter()
        .zip(&definition.hypergraph.adjacency)
        .filter(|(operation, _)| operation.as_str().starts_with(GENERATED_CONTEXT_PREFIX))
    {
        let raw = RawTheoryArrow {
            name: operation.clone(),
            type_maps: (
                boundary_to_hexpr(
                    &node_types(definition, &boundary.sources),
                    ambient_context_arity,
                ),
                boundary_to_hexpr(
                    &node_types(definition, &boundary.targets),
                    ambient_context_arity,
                ),
            ),
            definition: None,
        };
        let type_maps = interpret_type_maps(syntax, &raw.type_maps)?;
        arrows.insert(
            operation.clone(),
            TheoryArrow {
                name: operation.clone(),
                raw,
                type_maps,
                definition: None,
            },
        );
    }
    Ok(())
}

fn boundary_to_hexpr(objects: &[Obj], context_arity: usize) -> Hexpr {
    if objects.is_empty() {
        return Hexpr::Frobenius {
            sources: context_vars(context_arity),
            targets: vec![],
        };
    }
    let context = context_vars(context_arity);
    let mut leaves = Vec::new();
    for object in objects {
        collect_leaf_indices(object, &mut leaves);
    }
    Hexpr::Composition(vec![
        Hexpr::Frobenius {
            sources: context.clone(),
            targets: leaves
                .into_iter()
                .map(|leaf| context[leaf].clone())
                .collect(),
        },
        objects_to_hexpr(objects),
    ])
}

fn interpret_type_maps(
    syntax: &Theory,
    maps: &(Hexpr, Hexpr),
) -> Result<(Term, Term), ReplaceClosuresError> {
    let source = interpret_type_map(syntax, &maps.0)?;
    let target = interpret_type_map(syntax, &maps.1)?;
    if source.sources != target.sources {
        return Err(ReplaceClosuresError::TypeMapDomainMismatch);
    }
    Ok((source, target))
}

fn interpret_type_map(syntax: &Theory, map: &Hexpr) -> Result<Term, ReplaceClosuresError> {
    try_interpret(&syntax.local_signature(), map)
        .map(|term| term.map_nodes(|_| ()))
        .map_err(|error| ReplaceClosuresError::TypeMapInterpretation {
            map: map.clone(),
            error,
        })
}

fn interface_types(term: &Term) -> Result<Vec<Obj>, ReplaceClosuresError> {
    let mut term = term.clone();
    term.quotient().map_err(|error| {
        ReplaceClosuresError::TypeMapEvaluation(format!("could not quotient type map: {error:?}"))
    })?;
    let values = eval_type(
        term.clone()
            .map_edges(|operation| WithSpiders::Operation(Dual::Fwd(operation))),
    )
    .map_err(|error| ReplaceClosuresError::TypeMapEvaluation(format!("{error:?}")))?;
    let compact_by_source_node = term
        .sources
        .iter()
        .enumerate()
        .map(|(compact, node)| (node.0, compact))
        .collect::<BTreeMap<_, _>>();
    Ok(term
        .targets
        .iter()
        .map(|node| compact_type_map_leaves(&values[node.0], &compact_by_source_node))
        .collect::<Result<_, _>>()?)
}

fn compact_type_map_leaves(
    object: &Obj,
    compact_by_source_node: &BTreeMap<usize, usize>,
) -> Result<Obj, ReplaceClosuresError> {
    match object {
        Tree::Empty => Ok(Tree::Empty),
        Tree::Leaf(node, annotation) => compact_by_source_node
            .get(node)
            .copied()
            .map(|compact| Tree::Leaf(compact, *annotation))
            .ok_or(ReplaceClosuresError::TypeMapEvaluation(format!(
                "type-map target depends on non-context node w{node}"
            ))),
        Tree::Node(operation, annotation, children) => Ok(Tree::Node(
            operation.clone(),
            *annotation,
            children
                .iter()
                .map(|child| compact_type_map_leaves(child, compact_by_source_node))
                .collect::<Result<_, _>>()?,
        )),
    }
}

fn node_types(term: &TypedTerm, nodes: &[NodeId]) -> Vec<Obj> {
    nodes
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn instantiate_context(object: &Obj, originals: &[usize]) -> Result<Obj, ReplaceClosuresError> {
    match object {
        Tree::Empty => Ok(Tree::Empty),
        Tree::Leaf(local, annotation) => originals
            .get(*local)
            .copied()
            .map(|original| Tree::Leaf(original, *annotation))
            .ok_or(ReplaceClosuresError::MissingOriginalContextLeaf { leaf: *local }),
        Tree::Node(operation, annotation, children) => Ok(Tree::Node(
            operation.clone(),
            *annotation,
            children
                .iter()
                .map(|child| instantiate_context(child, originals))
                .collect::<Result<_, _>>()?,
        )),
    }
}

fn collect_leaf_indices(object: &Obj, leaves: &mut impl Extend<usize>) {
    match object {
        Tree::Empty => {}
        Tree::Leaf(index, _) => leaves.extend([*index]),
        Tree::Node(_, _, children) => {
            for child in children {
                collect_leaf_indices(child, leaves);
            }
        }
    }
}

fn name_operation(definition: &Operation, closure: NodeId) -> Operation {
    format!("{NAME_PREFIX}{}", closure_operation(definition, closure))
        .parse()
        .expect("generated name operation should parse")
}

fn context_operation(definition: &Operation, closure: NodeId) -> Operation {
    format!(
        "{GENERATED_CONTEXT_PREFIX}closure.{definition}.{}",
        closure.0
    )
    .parse()
    .expect("generated context operation should parse")
}

fn context_vars(arity: usize) -> Vec<Variable> {
    (0..arity)
        .map(|index| {
            format!("{GENERATED_VARIABLE_PREFIX}closure_ctx{index}")
                .parse()
                .expect("generated context variable should parse")
        })
        .collect()
}
