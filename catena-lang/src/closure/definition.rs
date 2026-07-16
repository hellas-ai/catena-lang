//! Create generated arrows from discovered closure regions.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use hexpr::{Hexpr, Operation, Variable, interpret::Error as HexprInterpretError, try_interpret};
use metacat::{
    theory::{
        Term, Theory, TheoryArrow, TheoryId, TheorySet, ast::RawTheoryArrow, model::SignatureError,
    },
    tree::Tree,
};
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    closure::region::{ClosureRegion, ClosureRegionMap},
    elaborate::{ElaborateError, name_symbols},
    hexpr::{objects_to_hexpr, term_to_hexpr},
    nonstrict::{to_packer, to_unpacker, unpack_packed_object},
    pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm},
    prefixes::{GENERATED_VARIABLE_PREFIX, NAME_PREFIX},
    report::TheoryTermMap,
};

type Obj = Tree<(), Operation>;

/// Compact generated-context leaves mapped back to leaves of the original
/// definition, keyed by generated `closure.*` operation.
pub type ClosureContextMap = BTreeMap<TheoryId, BTreeMap<Operation, Vec<usize>>>;

/// Generated declarations together with their still-annotated closure bodies.
#[derive(Debug, Clone)]
pub struct DefinedClosures {
    pub generated_theory: TheorySet,
    pub generated_functions: TheoryTermMap,
    pub closure_contexts: ClosureContextMap,
}

#[derive(Debug, Error)]
pub enum DefineClosuresError {
    #[error("missing theory `{0}`")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing syntax theory `{0}`")]
    MissingSyntaxTheory(String),
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("region node w{node} is out of bounds in `{theory}.{definition}`")]
    NodeOutOfBounds {
        theory: String,
        definition: String,
        node: usize,
    },
    #[error("region edge e{edge} is out of bounds in `{theory}.{definition}`")]
    EdgeOutOfBounds {
        theory: String,
        definition: String,
        edge: usize,
    },
    #[error("region body contains nested closure marker e{edge} in `{theory}.{definition}`")]
    NestedClosureMarker {
        theory: String,
        definition: String,
        edge: usize,
    },
    #[error("region edge e{edge} references a node outside its region")]
    IncidentNodeOutsideRegion { edge: usize },
    #[error("generated region contains `eval` e{edge} without a matching `name.*` producer")]
    EvalWithoutName { edge: usize },
    #[error("generated region references missing operation `{operation}`")]
    MissingOperation { operation: String },
    #[error("failed to quotient generated body for `{theory}.{definition}`: {error}")]
    Quotient {
        theory: String,
        definition: String,
        error: String,
    },
    #[error(transparent)]
    NameElaboration(#[from] ElaborateError),
    #[error("failed to interpret generated type map `{map}`: {error}")]
    TypeMapInterpretation {
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
    #[error("generated type maps have incompatible context domains")]
    TypeMapDomainMismatch,
}

/// Insert a `closure.*` definition and matching `name.closure.*` declaration
/// for every discovered region. Original definitions are left unchanged for
/// the later replacement stage.
pub fn run(
    theory_set: &TheorySet,
    forgotten: &TheoryTermMap<ClosureForgotten<Operation>>,
    regions: &ClosureRegionMap,
) -> Result<DefinedClosures, DefineClosuresError> {
    let mut generated_theory = theory_set.clone();
    let mut generated_functions = BTreeMap::new();
    let mut closure_contexts = BTreeMap::new();

    for (theory_id, definitions) in regions {
        let original_theory = theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| DefineClosuresError::MissingTheory(theory_id.to_string()))?;
        let Theory::Theory { syntax, arrows } = original_theory else {
            return Err(DefineClosuresError::NotUserTheory(theory_id.to_string()));
        };
        let syntax_theory = theory_set
            .theories
            .get(syntax)
            .ok_or_else(|| DefineClosuresError::MissingSyntaxTheory(syntax.to_string()))?;
        let forgotten_definitions = forgotten
            .get(theory_id)
            .ok_or_else(|| DefineClosuresError::MissingTheory(theory_id.to_string()))?;

        let mut generated_arrows = BTreeMap::new();
        let mut generated_bodies = BTreeMap::new();
        let mut generated_contexts = BTreeMap::new();
        for (definition_name, definition_regions) in definitions {
            if definition_regions.is_empty() {
                continue;
            }
            if !arrows.contains_key(definition_name) {
                return Err(DefineClosuresError::MissingDefinition {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                });
            }
            let term = forgotten_definitions.get(definition_name).ok_or_else(|| {
                DefineClosuresError::MissingDefinition {
                    theory: theory_id.to_string(),
                    definition: definition_name.to_string(),
                }
            })?;

            for region in definition_regions {
                let generated = define_region(
                    theory_id,
                    definition_name,
                    syntax_theory,
                    arrows,
                    term,
                    region,
                )?;
                let closure_name = generated.closure.name.clone();
                generated_contexts.insert(closure_name.clone(), generated.original_context_leaves);
                generated_bodies.insert(closure_name.clone(), generated.body);
                generated_arrows.insert(closure_name, generated.closure);
                generated_arrows.insert(generated.name.name.clone(), generated.name);
            }
        }

        let Theory::Theory { arrows, .. } = generated_theory
            .theories
            .get_mut(theory_id)
            .expect("validated theory should remain present")
        else {
            unreachable!("validated user theory should remain a user theory")
        };
        arrows.extend(generated_arrows);
        if !generated_bodies.is_empty() {
            generated_functions.insert(theory_id.clone(), generated_bodies);
            closure_contexts.insert(theory_id.clone(), generated_contexts);
        }
    }

    Ok(DefinedClosures {
        generated_theory,
        generated_functions,
        closure_contexts,
    })
}

/// The two arrows produced for one region and the annotated body of the
/// executable `closure.*` arrow.
struct GeneratedClosure {
    closure: TheoryArrow,
    name: TheoryArrow,
    body: AnnotatedTerm,
    original_context_leaves: Vec<usize>,
}

/// Turn one discovered region into:
///
/// - `closure.<definition>.<node>`, an executable arrow whose inputs are the
///   packed runtime environment and the original closure argument; and
/// - `name.closure.<definition>.<node>`, the corresponding function pointer.
fn define_region(
    theory_id: &TheoryId,
    definition_name: &Operation,
    syntax_theory: &Theory,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    term: &ClosureForgottenTerm,
    region: &ClosureRegion,
) -> Result<GeneratedClosure, DefineClosuresError> {
    let body = build_body(theory_id, definition_name, arrows, term, region)?;

    // Runtime captures are already represented by the body's first input.
    // This context is different: it contains type-level variables required by
    // any node in the generated body. Compact them so each generated definition
    // has a minimal, locally numbered type context, while retaining the reverse
    // mapping needed to wire `name.closure.*` at the original use site.
    let context = ClosureContext::from_term(&body);
    let original_context_leaves = context.original_leaf_by_compact_leaf.clone();
    let body = context.relabel_term(body);

    let closure_name = closure_operation(definition_name, region.closure);
    let raw_closure = RawTheoryArrow {
        name: closure_name.clone(),
        type_maps: type_maps_for_term(&body, context.arity()),
        definition: Some(term_to_hexpr(&body)),
    };
    let closure = TheoryArrow {
        name: closure_name,
        type_maps: interpret_type_maps(syntax_theory, &raw_closure.type_maps)?,
        definition: Some(body.clone().map_nodes(|_| ())),
        raw: raw_closure.clone(),
    };

    let raw_name = name_symbols::name_arrow(syntax_theory, &theory_id.0, &raw_closure)?;
    let name = TheoryArrow {
        name: raw_name.name.clone(),
        type_maps: interpret_type_maps(syntax_theory, &raw_name.type_maps)?,
        definition: None,
        raw: raw_name,
    };

    Ok(GeneratedClosure {
        closure,
        name,
        body,
        original_context_leaves,
    })
}

fn build_body(
    theory_id: &TheoryId,
    definition_name: &Operation,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    term: &ClosureForgottenTerm,
    region: &ClosureRegion,
) -> Result<AnnotatedTerm, DefineClosuresError> {
    let mut body = AnnotatedTerm::empty();
    let mut node_map = HashMap::new();

    // A `name.f -> eval` pair is the first-order encoding of a direct call to
    // `f`. Reconstruct that call in the generated function instead of copying
    // the temporary name and eval operations into its body.
    let named_evals = named_evals(term, region)?;
    let skipped_edges = named_evals
        .iter()
        .flat_map(|pair| [pair.name, pair.eval])
        .collect::<HashSet<_>>();

    // Region discovery retains useful control-flow nodes as well as the nodes
    // incident to its operations. A standalone function only needs its
    // boundary and the nodes used by operations that we copy or reconstruct.
    let mut required_nodes = HashSet::from([region.domain, region.codomain]);
    required_nodes.extend(region.environment.iter().copied());
    for pair in &named_evals {
        let eval = &term.hypergraph.adjacency[pair.eval.0];
        required_nodes.extend(eval.sources.first().copied());
        required_nodes.extend(eval.targets.first().copied());
    }
    for &edge in &region.edges {
        if skipped_edges.contains(&edge) {
            continue;
        }
        let boundary = &term.hypergraph.adjacency[edge.0];
        required_nodes.extend(boundary.sources.iter().copied());
        required_nodes.extend(boundary.targets.iter().copied());
    }

    // Copy the required nodes first, recording the old-to-new correspondence
    // used by copied edges and reconstructed calls below.
    for &node in region
        .nodes
        .iter()
        .filter(|node| required_nodes.contains(node))
    {
        let ty = term.hypergraph.nodes.get(node.0).cloned().ok_or_else(|| {
            DefineClosuresError::NodeOutOfBounds {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
                node: node.0,
            }
        })?;
        node_map.insert(node, body.new_node(ty));
    }

    // Copy ordinary region operations. Named eval pairs are handled separately.
    for &edge in &region.edges {
        if skipped_edges.contains(&edge) {
            continue;
        }
        let operation = term.hypergraph.edges.get(edge.0).ok_or_else(|| {
            DefineClosuresError::EdgeOutOfBounds {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
                edge: edge.0,
            }
        })?;
        let ClosureForgotten::Operation(operation) = operation else {
            return Err(DefineClosuresError::NestedClosureMarker {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
                edge: edge.0,
            });
        };
        let boundary = &term.hypergraph.adjacency[edge.0];
        let sources = remap_nodes(&node_map, edge, &boundary.sources)?;
        let targets = remap_nodes(&node_map, edge, &boundary.targets)?;
        body.new_edge(operation.clone(), (sources, targets));
    }

    for pair in named_evals {
        inline_named_eval(&mut body, &node_map, arrows, term, pair)?;
    }

    // Initially expose every captured value separately. Packing them creates a
    // single runtime environment input, including the unit object when there
    // are no captures. The second input and sole output are the marker's domain
    // and codomain respectively.
    body.sources = remap_nodes(&node_map, region.marker, &region.environment)?;
    let domain =
        *node_map
            .get(&region.domain)
            .ok_or_else(|| DefineClosuresError::NodeOutOfBounds {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
                node: region.domain.0,
            })?;
    let codomain =
        *node_map
            .get(&region.codomain)
            .ok_or_else(|| DefineClosuresError::NodeOutOfBounds {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
                node: region.codomain.0,
            })?;
    let environment = packed_environment_source(&mut body);
    body.sources = vec![environment, domain];
    body.targets = vec![codomain];
    body.quotient()
        .map_err(|error| DefineClosuresError::Quotient {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
            error: format!("{error:?}"),
        })?;
    Ok(body)
}

#[derive(Debug, Clone, Copy)]
struct NamedEval {
    name: EdgeId,
    eval: EdgeId,
}

fn named_evals(
    term: &ClosureForgottenTerm,
    region: &ClosureRegion,
) -> Result<Vec<NamedEval>, DefineClosuresError> {
    let included = region.edges.iter().copied().collect::<HashSet<_>>();
    let mut producers = vec![Vec::new(); term.hypergraph.nodes.len()];
    for &edge in &region.edges {
        for &target in &term.hypergraph.adjacency[edge.0].targets {
            producers[target.0].push(edge);
        }
    }

    region
        .edges
        .iter()
        .copied()
        .filter(|edge| {
            matches!(
                &term.hypergraph.edges[edge.0],
                ClosureForgotten::Operation(operation) if operation.as_str() == "eval"
            )
        })
        .map(|eval| {
            let boundary = &term.hypergraph.adjacency[eval.0];
            let pointer = boundary
                .sources
                .get(1)
                .copied()
                .ok_or(DefineClosuresError::EvalWithoutName { edge: eval.0 })?;
            let name = producers[pointer.0]
                .iter()
                .copied()
                .find(|producer| {
                    included.contains(producer)
                        && matches!(
                            &term.hypergraph.edges[producer.0],
                            ClosureForgotten::Operation(operation)
                                if operation.as_str().starts_with(NAME_PREFIX)
                        )
                })
                .ok_or(DefineClosuresError::EvalWithoutName { edge: eval.0 })?;
            Ok(NamedEval { name, eval })
        })
        .collect()
}

fn inline_named_eval(
    body: &mut AnnotatedTerm,
    node_map: &HashMap<NodeId, NodeId>,
    arrows: &BTreeMap<Operation, TheoryArrow>,
    term: &ClosureForgottenTerm,
    pair: NamedEval,
) -> Result<(), DefineClosuresError> {
    let ClosureForgotten::Operation(name_operation) = &term.hypergraph.edges[pair.name.0] else {
        unreachable!("named eval producer should be an operation");
    };
    let operation: Operation = name_operation
        .as_str()
        .strip_prefix(NAME_PREFIX)
        .expect("named eval producer should start with name prefix")
        .parse()
        .expect("stripped generated name should parse");
    let arrow = arrows
        .get(&operation)
        .ok_or_else(|| DefineClosuresError::MissingOperation {
            operation: operation.to_string(),
        })?;
    let eval = &term.hypergraph.adjacency[pair.eval.0];
    let ([domain, _pointer], [codomain]) = (eval.sources.as_slice(), eval.targets.as_slice())
    else {
        return Err(DefineClosuresError::EvalWithoutName { edge: pair.eval.0 });
    };
    let domain_type = term.hypergraph.nodes[domain.0].clone();
    let codomain_type = term.hypergraph.nodes[codomain.0].clone();
    let operation_sources = unpack_packed_object(&domain_type, arrow.type_maps.0.targets.len());
    let operation_targets = unpack_packed_object(&codomain_type, arrow.type_maps.1.targets.len());

    let unpacker = to_unpacker(operation_sources.clone());
    let (unpacker_sources, operation_inputs) = body.append(unpacker);
    let [unpacker_source] = unpacker_sources.as_slice() else {
        unreachable!("unpacker should have one packed source");
    };
    body.unify(*unpacker_source, node_map[domain]);

    let operation_outputs = operation_targets
        .iter()
        .map(|ty| body.new_node(ty.clone()))
        .collect::<Vec<_>>();
    body.new_edge(operation, (operation_inputs, operation_outputs.clone()));

    let packer = to_packer(operation_targets);
    let (packer_sources, packer_targets) = body.append(packer);
    for (output, packer_source) in operation_outputs.into_iter().zip(packer_sources) {
        body.unify(output, packer_source);
    }
    let [packer_target] = packer_targets.as_slice() else {
        unreachable!("packer should have one packed target");
    };
    body.unify(*packer_target, node_map[codomain]);
    Ok(())
}

fn remap_nodes(
    node_map: &HashMap<NodeId, NodeId>,
    edge: EdgeId,
    nodes: &[NodeId],
) -> Result<Vec<NodeId>, DefineClosuresError> {
    nodes
        .iter()
        .map(|node| {
            node_map
                .get(node)
                .copied()
                .ok_or(DefineClosuresError::IncidentNodeOutsideRegion { edge: edge.0 })
        })
        .collect()
}

fn packed_environment_source(body: &mut AnnotatedTerm) -> NodeId {
    let components = body.sources.clone();
    let component_types = components
        .iter()
        .map(|node| body.hypergraph.nodes[node.0].clone())
        .collect();
    let unpacker = to_unpacker(component_types);
    let (sources, targets) = body.append(unpacker);
    let [source] = sources.as_slice() else {
        unreachable!("one packed environment object should produce one source");
    };
    for (target, component) in targets.into_iter().zip(components) {
        body.unify(target, component);
    }
    *source
}

#[derive(Debug)]
struct ClosureContext {
    original_leaf_by_compact_leaf: Vec<usize>,
}

impl ClosureContext {
    /// Collect every type metavariable needed to check the generated body.
    ///
    /// Looking only at public inputs and outputs is insufficient: an operation
    /// inside the closure may use a type metavariable which is absent from the
    /// runtime boundary. That metavariable must still be part of the generated
    /// `closure.*` and `name.closure.*` context.
    fn from_term(term: &AnnotatedTerm) -> Self {
        let mut leaves = BTreeSet::new();
        for object in &term.hypergraph.nodes {
            collect_leaf_indices(object, &mut leaves);
        }
        Self {
            original_leaf_by_compact_leaf: leaves.into_iter().collect(),
        }
    }

    fn arity(&self) -> usize {
        self.original_leaf_by_compact_leaf.len()
    }

    fn relabel_term(&self, term: AnnotatedTerm) -> AnnotatedTerm {
        let compact = self
            .original_leaf_by_compact_leaf
            .iter()
            .copied()
            .enumerate()
            .map(|(local, original)| (original, local))
            .collect::<BTreeMap<_, _>>();
        term.map_nodes(|object| relabel_object_context(object, &compact))
    }
}

fn relabel_object_context(object: Obj, compact: &BTreeMap<usize, usize>) -> Obj {
    match object {
        Tree::Empty => Tree::Empty,
        Tree::Leaf(original, annotation) => Tree::Leaf(
            compact.get(&original).copied().unwrap_or(original),
            annotation,
        ),
        Tree::Node(operation, annotation, children) => Tree::Node(
            operation,
            annotation,
            children
                .into_iter()
                .map(|child| relabel_object_context(child, compact))
                .collect(),
        ),
    }
}

fn type_maps_for_term(term: &AnnotatedTerm, context_arity: usize) -> (Hexpr, Hexpr) {
    (
        objects_to_hexpr_in_context(&interface_types(term, &term.sources), context_arity),
        objects_to_hexpr_in_context(&interface_types(term, &term.targets), context_arity),
    )
}

fn objects_to_hexpr_in_context(objects: &[Obj], context_arity: usize) -> Hexpr {
    let context = context_vars(context_arity);
    let used = leaf_indices(objects)
        .into_iter()
        .map(|leaf| context[leaf].clone())
        .collect();
    Hexpr::Composition(vec![
        Hexpr::Frobenius {
            sources: context,
            targets: used,
        },
        objects_to_hexpr(objects),
    ])
}

fn interpret_type_maps(
    syntax: &Theory,
    maps: &(Hexpr, Hexpr),
) -> Result<(Term, Term), DefineClosuresError> {
    let source = interpret_type_map(syntax, &maps.0)?;
    let target = interpret_type_map(syntax, &maps.1)?;
    if source.sources != target.sources {
        return Err(DefineClosuresError::TypeMapDomainMismatch);
    }
    Ok((source, target))
}

fn interpret_type_map(syntax: &Theory, map: &Hexpr) -> Result<Term, DefineClosuresError> {
    try_interpret(&syntax.local_signature(), map)
        .map(|term| term.map_nodes(|_| ()))
        .map_err(|error| DefineClosuresError::TypeMapInterpretation {
            map: map.clone(),
            error,
        })
}

pub(crate) fn closure_operation(definition: &Operation, closure: NodeId) -> Operation {
    format!("closure.{definition}.{}", closure.0)
        .parse()
        .expect("generated closure operation should parse")
}

fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn leaf_indices(objects: &[Obj]) -> Vec<usize> {
    let mut indices = Vec::new();
    for object in objects {
        collect_leaf_indices(object, &mut indices);
    }
    indices
}

fn collect_leaf_indices(object: &Obj, indices: &mut impl Extend<usize>) {
    match object {
        Tree::Empty => {}
        Tree::Leaf(index, _) => indices.extend([*index]),
        Tree::Node(_, _, children) => {
            for child in children {
                collect_leaf_indices(child, indices);
            }
        }
    }
}

fn context_vars(arity: usize) -> Vec<Variable> {
    (0..arity)
        .map(|index| {
            format!("{GENERATED_VARIABLE_PREFIX}closure_ctx{index}")
                .parse()
                .expect("generated variable should parse")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closure_context_includes_leaves_used_only_inside_body() {
        let mut body = AnnotatedTerm::empty();
        let source = body.new_node(Tree::Empty);
        let target = body.new_node(Tree::Empty);
        body.new_node(Tree::Leaf(4, ())); // Not part of the public boundary.
        body.sources = vec![source];
        body.targets = vec![target];

        let context = ClosureContext::from_term(&body);

        assert_eq!(context.original_leaf_by_compact_leaf, vec![4]);
    }

    #[test]
    fn target_only_dependency_uses_one_shared_type_map_domain() {
        let object = |name: &str, children: Vec<Obj>| {
            Tree::Node(
                name.parse().expect("test operation should parse"),
                0,
                children,
            )
        };
        let mut body = AnnotatedTerm::empty();
        let source = body.new_node(object("val", vec![object("u64", vec![])]));
        let target = body.new_node(object("val", vec![object("ix", vec![Tree::Leaf(4, ())])]));
        body.sources = vec![source];
        body.targets = vec![target];

        let context = ClosureContext::from_term(&body);
        assert_eq!(context.original_leaf_by_compact_leaf, vec![4]);
        let body = context.relabel_term(body);
        let maps = type_maps_for_term(&body, context.arity());

        let raw = metacat::theory::RawTheorySet::from_texts(crate::stdlib::sources())
            .expect("standard library should parse");
        let elaborated =
            crate::elaborate::elaborate(raw).expect("standard library should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("standard library should load");
        let program = TheoryId("program".parse().unwrap());
        let Theory::Theory { syntax, .. } = &theory_set.theories[&program] else {
            panic!("program should be a user theory");
        };
        let syntax = &theory_set.theories[syntax];
        let (source_map, target_map) = interpret_type_maps(syntax, &maps).unwrap();

        assert_eq!(source_map.sources, target_map.sources);
        assert_eq!(source_map.sources.len(), 1);
    }
}
