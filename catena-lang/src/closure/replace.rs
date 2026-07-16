//! Replace discovered closure regions with explicit environments and function pointers.

use std::collections::{BTreeMap, BTreeSet};

use hexpr::{Hexpr, Operation, Variable, interpret::Error as HexprInterpretError, try_interpret};
use metacat::{
    check::eval_type,
    dual::Dual,
    spiders::WithSpiders,
    theory::{
        Term, Theory, TheoryArrow, TheoryId, TheorySet, ast::RawTheoryArrow, model::SignatureError,
    },
    tree::Tree,
};
use open_hypergraphs::lax::{EdgeId, Hyperedge, NodeId};
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    closure::{
        definition::{ClosureContextMap, closure_operation},
        region::{ClosureRegion, ClosureRegionMap, find_regions},
    },
    hexpr::{objects_to_hexpr, term_to_hexpr},
    nonstrict::to_packer,
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
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
    #[error("missing context mapping for generated closure `{operation}`")]
    MissingClosureContext { operation: String },
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
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
    generated_functions: &TheoryTermMap,
    regions: &ClosureRegionMap,
    closure_contexts: &ClosureContextMap,
) -> Result<Replacement, ReplaceClosuresError> {
    let mut output = theory_set.clone();
    // Generated closure bodies are already ordinary operation graphs. Keep
    // them separate from marker-bearing forgotten definitions and include them
    // directly in the final definition map.
    let mut terms = generated_functions.clone();

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
            let rewritten = replace_definition_regions(
                theory_id,
                definition_name,
                arrows,
                term,
                definition_regions,
                closure_contexts,
            )?;

            // Context projections are temporary graph operations introduced by
            // replacement. Declare them so the rewritten theory can be checked;
            // the final closure-conversion step erases them from runtime code.
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
        terms
            .entry(theory_id.clone())
            .or_default()
            .extend(replaced_definitions);
    }

    Ok(Replacement {
        theory_set: output,
        terms,
    })
}

/// Replace every region in one definition, then turn the marker-bearing graph
/// back into an ordinary annotated definition.
fn replace_definition_regions(
    theory_id: &TheoryId,
    definition_name: &Operation,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    definition: &ClosureForgottenTerm,
    original_regions: &[ClosureRegion],
    closure_contexts: &ClosureContextMap,
) -> Result<AnnotatedTerm, ReplaceClosuresError> {
    let mut rewritten = definition.clone();

    // Splicing changes node and edge identifiers. Rediscover the first
    // remaining region before every rewrite, but use `original_region` to keep
    // the stable generated closure name and context mapping.
    for original_region in original_regions {
        let generated_closure = closure_operation(definition_name, original_region.closure);
        let original_context_leaves = closure_contexts
            .get(theory_id)
            .and_then(|contexts| contexts.get(&generated_closure))
            .ok_or_else(|| ReplaceClosuresError::MissingClosureContext {
                operation: generated_closure.to_string(),
            })?;
        let current_region = first_remaining_region(theory_id, definition_name, &rewritten)?;
        let replacement = build_closure_value(
            definition_name,
            arrows,
            &rewritten,
            &current_region,
            original_region,
            original_context_leaves,
        )?;
        rewritten = splice_region(&rewritten, &current_region, &replacement)?;
    }

    if !find_regions(&rewritten)
        .map_err(|_| ReplaceClosuresError::RemainingClosureMarker)?
        .is_empty()
    {
        return Err(region_count_changed(theory_id, definition_name));
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
    Ok(rewritten)
}

fn first_remaining_region(
    theory_id: &TheoryId,
    definition_name: &Operation,
    definition: &ClosureForgottenTerm,
) -> Result<ClosureRegion, ReplaceClosuresError> {
    find_regions(definition)
        .ok()
        .and_then(|regions| regions.into_iter().next())
        .ok_or_else(|| region_count_changed(theory_id, definition_name))
}

fn region_count_changed(theory_id: &TheoryId, definition_name: &Operation) -> ReplaceClosuresError {
    ReplaceClosuresError::RegionCountChanged {
        theory: theory_id.to_string(),
        definition: definition_name.to_string(),
    }
}

/// Build the value which replaces one opaque closure node.
///
/// Before replacement, the marker delimits the body and produces one opaque
/// closure value `C`:
///
/// ```text
/// captures ────────> [ closure body ]
/// A ───────────────> [              ] ──> B
/// A, B ────────────> !closure ──────────> C
/// ```
///
/// The body has already been cut out as `closure.*`. This function constructs
/// the two values which replace `C`:
///
/// ```text
/// captures ─> context.closure.* ─> copies ─> pack ─> Environment
///                         `──────> context ─> name.closure.* ─> FnPointer
/// ```
///
/// The context operation temporarily exposes both linear capture copies and
/// the original type-level context required by `name.closure.*`. It is erased
/// at the end of closure conversion.
fn build_closure_value(
    definition_name: &Operation,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    definition: &ClosureForgottenTerm,
    region: &ClosureRegion,
    original_region: &ClosureRegion,
    original_context_leaves: &[usize],
) -> Result<ClosureForgottenTerm, ReplaceClosuresError> {
    let generated_name = generated_name_boundary(
        definition_name,
        arrows,
        original_region.closure,
        original_context_leaves,
    )?;

    let mut replacement = ClosureForgottenTerm::empty();
    let capture_sources = region
        .environment
        .iter()
        .map(|node| {
            definition
                .hypergraph
                .nodes
                .get(node.0)
                .cloned()
                .map(|object| replacement.new_node(object))
                .ok_or(ReplaceClosuresError::NodeOutOfBounds { node: node.0 })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let projected = add_context_projection(
        &mut replacement,
        definition_name,
        original_region.closure,
        &capture_sources,
        &generated_name.source_types,
    );
    let environment = pack_environment(&mut replacement, projected.environment_components);
    let function_pointer =
        add_generated_name(&mut replacement, generated_name, projected.name_sources);

    replacement.sources = capture_sources;
    replacement.targets = vec![environment, function_pointer];
    Ok(replacement)
}

struct GeneratedNameBoundary {
    operation: Operation,
    source_types: Vec<Obj>,
    target_type: Obj,
}

/// Read the generated declaration and translate its compact context leaves back
/// to the context of the original definition.
fn generated_name_boundary(
    definition_name: &Operation,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    original_closure: NodeId,
    original_context_leaves: &[usize],
) -> Result<GeneratedNameBoundary, ReplaceClosuresError> {
    let operation = name_operation(definition_name, original_closure);
    let arrow =
        arrows
            .get(&operation)
            .ok_or_else(|| ReplaceClosuresError::MissingNameOperation {
                operation: operation.to_string(),
            })?;
    let source_types = interface_types(&arrow.type_maps.0)?
        .iter()
        .map(|object| instantiate_context(object, original_context_leaves))
        .collect::<Result<_, _>>()?;
    let target_types = interface_types(&arrow.type_maps.1)?;
    let [target_type] = target_types.as_slice() else {
        return Err(ReplaceClosuresError::InvalidNameTargets {
            operation: operation.to_string(),
            targets: target_types.len(),
        });
    };

    Ok(GeneratedNameBoundary {
        operation,
        source_types,
        target_type: instantiate_context(target_type, original_context_leaves)?,
    })
}

struct ProjectedInputs {
    environment_components: Vec<NodeId>,
    name_sources: Vec<NodeId>,
}

/// Add the temporary context projection which makes fresh linear copies of the
/// captures and exposes the static inputs of `name.closure.*`.
fn add_context_projection(
    replacement: &mut ClosureForgottenTerm,
    definition_name: &Operation,
    original_closure: NodeId,
    capture_sources: &[NodeId],
    name_source_types: &[Obj],
) -> ProjectedInputs {
    let environment_components = capture_sources
        .iter()
        .map(|source| replacement.new_node(replacement.hypergraph.nodes[source.0].clone()))
        .collect::<Vec<_>>();
    let name_sources = name_source_types
        .iter()
        .map(|object| replacement.new_node(object.clone()))
        .collect::<Vec<_>>();
    let targets = environment_components
        .iter()
        .chain(&name_sources)
        .copied()
        .collect::<Vec<_>>();
    replacement.new_edge(
        ClosureForgotten::Operation(context_operation(definition_name, original_closure)),
        (capture_sources.to_vec(), targets),
    );
    ProjectedInputs {
        environment_components,
        name_sources,
    }
}

fn pack_environment(replacement: &mut ClosureForgottenTerm, components: Vec<NodeId>) -> NodeId {
    let component_types = components
        .iter()
        .map(|node| replacement.hypergraph.nodes[node.0].clone())
        .collect();
    let packer = to_packer(component_types).map_edges(ClosureForgotten::Operation);
    let (packer_sources, packer_targets) = replacement.append(packer);
    for (component, packer_source) in components.into_iter().zip(packer_sources) {
        replacement.unify(component, packer_source);
    }
    let [environment] = packer_targets.as_slice() else {
        unreachable!("environment packer should have one target")
    };
    *environment
}

fn add_generated_name(
    replacement: &mut ClosureForgottenTerm,
    generated_name: GeneratedNameBoundary,
    sources: Vec<NodeId>,
) -> NodeId {
    let function_pointer = replacement.new_node(generated_name.target_type);
    replacement.new_edge(
        ClosureForgotten::Operation(generated_name.operation),
        (sources, vec![function_pointer]),
    );
    function_pointer
}

/// Splice `(Environment, FnPointer)` in place of the closure node.
///
/// Every old incidence of the single closure node expands in order:
///
/// ```text
/// ... C ...    becomes    ... Environment, FnPointer ...
/// ```
///
/// This is why retained edge boundaries are rebuilt after deleting the region:
/// a normal node remap maps one node to one node, while closure replacement maps
/// one node to two nodes and therefore changes consumer arity.
fn splice_region(
    definition: &ClosureForgottenTerm,
    region: &ClosureRegion,
    replacement: &ClosureForgottenTerm,
) -> Result<ClosureForgottenTerm, ReplaceClosuresError> {
    assert_replacement_boundary(definition, region, replacement);
    let deletion = plan_region_deletion(definition, region)?;

    // First remove the old body, marker, and opaque closure node. Environment
    // nodes survive because they become the inputs of the replacement graph.
    let mut rewritten = definition.clone();
    rewritten.delete_edges(&deletion.edges);
    let node_map = rewritten.hypergraph.delete_nodes_witness(&deletion.nodes);
    rewritten.sources = expand_closure_node(&node_map, &definition.sources, region.closure, &[])?;

    // Append the small graph built by `build_closure_value` and identify its
    // capture inputs with the retained environment nodes of the outer graph.
    let retained_environment = region
        .environment
        .iter()
        .map(|node| remap_node(&node_map, *node))
        .collect::<Result<Vec<_>, _>>()?;
    let (replacement_sources, replacement_targets) = rewritten.append(replacement.clone());
    for (outer, inner) in retained_environment.into_iter().zip(replacement_sources) {
        rewritten.unify(outer, inner);
    }

    // Deleting nodes gives a one-to-one node map. Rebuild retained boundaries
    // explicitly so an occurrence of the old closure node can instead expand
    // to both replacement outputs `(Environment, FnPointer)`.
    for (new_edge, old_edge) in deletion.retained_edges.iter().enumerate() {
        let old = &definition.hypergraph.adjacency[old_edge.0];
        rewritten.hypergraph.adjacency[new_edge] = Hyperedge {
            sources: expand_closure_node(
                &node_map,
                &old.sources,
                region.closure,
                &replacement_targets,
            )?,
            targets: expand_closure_node(
                &node_map,
                &old.targets,
                region.closure,
                &replacement_targets,
            )?,
        };
    }
    rewritten.targets = expand_closure_node(
        &node_map,
        &definition.targets,
        region.closure,
        &replacement_targets,
    )?;
    Ok(rewritten)
}

struct RegionDeletion {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    retained_edges: Vec<EdgeId>,
}

fn assert_replacement_boundary(
    definition: &ClosureForgottenTerm,
    region: &ClosureRegion,
    replacement: &ClosureForgottenTerm,
) {
    assert!(
        replacement.clone().to_strict().is_monogamous(),
        "the constructed closure replacement must be monogamous"
    );
    assert_eq!(
        replacement.sources.len(),
        region.environment.len(),
        "closure replacement source arity must match the captured environment"
    );
    assert_eq!(
        replacement.targets.len(),
        2,
        "closure replacement must produce exactly (Environment, FnPointer)"
    );
    for (index, (&environment, &replacement_source)) in region
        .environment
        .iter()
        .zip(&replacement.sources)
        .enumerate()
    {
        let environment_type = &definition.hypergraph.nodes[environment.0];
        let replacement_type = &replacement.hypergraph.nodes[replacement_source.0];
        assert_eq!(
            environment_type, replacement_type,
            "closure replacement source {index} must have the same type as its captured environment wire"
        );
    }
}

/// Compute everything removed by a splice before mutating node and edge IDs.
fn plan_region_deletion(
    definition: &ClosureForgottenTerm,
    region: &ClosureRegion,
) -> Result<RegionDeletion, ReplaceClosuresError> {
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
        // The closure node is deliberately excluded: its incident consumer
        // edges survive and are expanded to consume `(Environment, FnPointer)`.
        .filter(|node| **node != region.closure)
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();

    let mut deleted_edges = region.edges.clone();
    deleted_edges.push(region.marker);
    deleted_edges.sort_by_key(|edge| edge.0);
    deleted_edges.dedup();
    let deleted_edge_set = deleted_edges
        .iter()
        .map(|edge| edge.0)
        .collect::<BTreeSet<_>>();

    // An operation outside the discovered region must never use an internal
    // node. Deleting such an operation would silently discard an escaping use;
    // keep this as a hard invariant of regions produced from CMC graphs.
    // Other `!closure` markers are metadata which may share control-flow
    // endpoints and are rediscovered after this splice.
    for (index, boundary) in definition.hypergraph.adjacency.iter().enumerate() {
        if deleted_edge_set.contains(&index)
            || matches!(
                definition.hypergraph.edges[index],
                ClosureForgotten::ClosureMarker
            )
        {
            continue;
        }
        let escaping_nodes = boundary
            .sources
            .iter()
            .chain(&boundary.targets)
            .filter(|node| deleted_node_set.contains(&node.0))
            .map(|node| node.0)
            .collect::<Vec<_>>();
        assert!(
            escaping_nodes.is_empty(),
            "closure region has escaping internal wires {escaping_nodes:?} at retained edge e{index} (`{}`: {:?} -> {:?}); domain=w{}, codomain=w{}",
            definition.hypergraph.edges[index],
            boundary.sources,
            boundary.targets,
            region.domain.0,
            region.codomain.0,
        );
    }

    let retained_edges = (0..definition.hypergraph.edges.len())
        .filter(|edge| !deleted_edge_set.contains(edge))
        .map(EdgeId)
        .collect::<Vec<_>>();

    Ok(RegionDeletion {
        nodes: deleted_nodes,
        edges: deleted_edges,
        retained_edges,
    })
}

fn expand_closure_node(
    node_map: &[Option<usize>],
    nodes: &[NodeId],
    closure: NodeId,
    closure_value: &[NodeId],
) -> Result<Vec<NodeId>, ReplaceClosuresError> {
    let mut output = Vec::new();
    for &node in nodes {
        if node == closure {
            output.extend_from_slice(closure_value);
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

fn unwrap_operations(term: ClosureForgottenTerm) -> Result<AnnotatedTerm, ReplaceClosuresError> {
    if term
        .hypergraph
        .edges
        .iter()
        .any(|edge| matches!(edge, ClosureForgotten::ClosureMarker))
    {
        return Err(ReplaceClosuresError::RemainingClosureMarker);
    }
    Ok(term.map_edges(|edge| match edge {
        ClosureForgotten::Operation(operation) => operation,
        ClosureForgotten::ClosureMarker => unreachable!("checked above"),
    }))
}

fn rewrite_converted_primitives(term: &mut AnnotatedTerm) {
    // Splicing expands each closure input:
    //
    //     primitive(..., Closure, ...)
    //
    // becomes
    //
    //     primitivec(..., Environment, FnPointer, ...)
    //
    // The graph boundary already has the expanded arity; select the matching
    // runtime primitive after all regions have been replaced.
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
    definition: &AnnotatedTerm,
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

fn node_types(term: &AnnotatedTerm, nodes: &[NodeId]) -> Vec<Obj> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "closure region has escaping internal wires")]
    fn deletion_plan_rejects_escaping_internal_wire() {
        let mut definition = ClosureForgottenTerm::empty();
        let domain = definition.new_node(Tree::Empty);
        let internal = definition.new_node(Tree::Empty);
        let codomain = definition.new_node(Tree::Empty);
        let closure = definition.new_node(Tree::Empty);
        let escaped = definition.new_node(Tree::Empty);
        let first = definition.new_edge(
            ClosureForgotten::Operation("first".parse().unwrap()),
            (vec![domain], vec![internal]),
        );
        let second = definition.new_edge(
            ClosureForgotten::Operation("second".parse().unwrap()),
            (vec![internal], vec![codomain]),
        );
        definition.new_edge(
            ClosureForgotten::Operation("outside".parse().unwrap()),
            (vec![internal], vec![escaped]),
        );
        let marker = definition.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![closure]),
        );
        let region = ClosureRegion {
            marker,
            domain,
            codomain,
            closure,
            environment: vec![],
            nodes: vec![domain, internal, codomain],
            edges: vec![first, second],
        };

        let _ = plan_region_deletion(&definition, &region);
    }
}
