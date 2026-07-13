use std::collections::{BTreeMap, BTreeSet};

use hexpr::{Hexpr, Operation, Variable, interpret::Error as HexprInterpretError, try_interpret};
use metacat::theory::{
    Term, Theory, TheoryArrow, TheoryId, TheorySet, ast::RawTheoryArrow, model::SignatureError,
};
use metacat::tree::Tree;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    check::{AnnotatedTerm, DefinitionTypes},
    closure::convert::{ConvertError, Converted, ConvertedClosure, convert},
    elaborate::{ElaborateError, name_symbols},
    hexpr::{objects_to_hexpr, term_to_hexpr},
    prefixes::{GENERATED_CONTEXT_PREFIX, GENERATED_VARIABLE_PREFIX},
    stdlib::constants::FN_HOM_TYPE,
};

const CONVERTED_PRIMITIVES: &[(&str, &str)] = &[
    ("if", "ifc"),
    ("bool.if", "bool.ifc"),
    ("reduce", "reducec"),
];

type Obj = Tree<(), Operation>;

struct GeneratedClosureNameDeclaration {
    operation: Operation,
    declared_sources: usize,
    declaration_source_type_map: Hexpr,
}

#[derive(Debug, Error)]
pub enum ConvertTheoryError {
    #[error("missing theory `{0}`")]
    MissingTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("missing syntax theory `{0}`")]
    MissingSyntaxTheory(String),
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("missing checked node types for definition `{definition}` in theory `{theory}`")]
    MissingDefinitionTypes { theory: String, definition: String },
    #[error(
        "typechecked node label count mismatch for definition `{definition}` in theory `{theory}`"
    )]
    NodeLabelCountMismatch { theory: String, definition: String },
    #[error(transparent)]
    Convert(#[from] ConvertError),
    #[error(transparent)]
    NameElaboration(#[from] ElaborateError),
    #[error("failed to interpret generated type map `{map}`: {error}")]
    TypeMapInterpretation {
        map: Hexpr,
        error: HexprInterpretError<SignatureError>,
    },
    #[error("generated type maps have incompatible domains")]
    TypeMapDomainMismatch,
}

pub fn convert_theory(
    theory_set: &TheorySet,
    definition_types: &DefinitionTypes,
    theory_id: &TheoryId,
) -> Result<Theory, ConvertTheoryError> {
    let theory = theory_set
        .theories
        .get(theory_id)
        .ok_or_else(|| ConvertTheoryError::MissingTheory(theory_id.to_string()))?;
    let Theory::Theory { syntax, arrows } = theory else {
        return Err(ConvertTheoryError::NotUserTheory(theory_id.to_string()));
    };
    let syntax_theory = theory_set
        .theories
        .get(syntax)
        .ok_or_else(|| ConvertTheoryError::MissingSyntaxTheory(syntax.to_string()))?;
    let theory_definition_types = definition_types.get(theory_id);

    // Region construction assumes closure-typed global interfaces have already
    // been inlined, so every closure passed to a converted primitive is built
    // locally from `defer`, `name.*`, and CMC closure combinators. If a helper
    // returning a closure is still a global call here, the region boundary is no
    // longer delimited by `defer`/`name.*` leaves and replacement wiring cannot
    // recover the required environment/name inputs.
    assert_closure_global_interfaces_are_inlined(theory_id, arrows);

    let mut converted_arrows = arrows.clone();
    for (definition_name, arrow) in arrows {
        if arrow.definition.is_none() {
            continue;
        }

        let typed = typed_definition(theory_id, definition_name, arrow, theory_definition_types)?;
        let closure_wires = primitive_closure_wires(&typed);
        if closure_wires.is_empty() {
            continue;
        }

        let converted = convert(definition_name, &typed, &closure_wires)?;
        update_definition_arrow(
            syntax_theory,
            theory_id,
            &mut converted_arrows,
            definition_name,
            arrow,
            converted,
        )?;
    }

    Ok(Theory::Theory {
        syntax: syntax.clone(),
        arrows: converted_arrows,
    })
}

fn assert_closure_global_interfaces_are_inlined(
    theory_id: &TheoryId,
    arrows: &BTreeMap<Operation, TheoryArrow>,
) {
    for (definition_name, arrow) in arrows {
        if arrow.definition.is_none() {
            continue;
        }

        assert!(
            !contains_closure_type_map(&arrow.type_maps.0)
                && !contains_closure_type_map(&arrow.type_maps.1),
            "closure conversion requires closure-boundary definitions to be inlined first; `{theory_id}.{definition_name}` still has a closure on its global interface"
        );
    }
}

fn update_definition_arrow(
    syntax: &Theory,
    theory_id: &TheoryId,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition_name: &Operation,
    original: &TheoryArrow,
    converted: Converted,
) -> Result<(), ConvertTheoryError> {
    let mut converted_definition = converted.definition;
    rewrite_converted_primitives(&mut converted_definition);

    let mut raw = original.raw.clone();
    raw.definition = Some(term_to_hexpr(&converted_definition));
    assert_eq!(
        original.type_maps.0.sources.len(),
        original.type_maps.1.sources.len(),
        "closure conversion expects original arrow type maps to share one context"
    );
    let ambient_context_arity = original.type_maps.0.sources.len();

    declare_context_arrows_from_use_sites(
        syntax,
        arrows,
        &converted_definition,
        ambient_context_arity,
    )?;
    let mut arrow = original.clone();
    arrow.raw = raw;
    arrow.definition = Some(converted_definition.clone().map_nodes(|_| ()));
    arrows.insert(definition_name.clone(), arrow);

    for closure in converted.closures {
        let name_boundary =
            insert_closure_arrows(syntax, theory_id, arrows, definition_name, closure)?;
        assert_generated_closure_name_use_matches_declaration(
            theory_id,
            definition_name,
            &converted_definition,
            &name_boundary,
        );
    }

    Ok(())
}

fn assert_generated_closure_name_use_matches_declaration(
    theory_id: &TheoryId,
    definition_name: &Operation,
    definition: &AnnotatedTerm,
    declaration: &GeneratedClosureNameDeclaration,
) {
    for (edge_index, (operation, edge)) in definition
        .hypergraph
        .edges
        .iter()
        .zip(&definition.hypergraph.adjacency)
        .enumerate()
        .filter(|(_, (operation, _))| *operation == &declaration.operation)
    {
        let connected_sources = edge.sources.len();
        let declared_sources = declaration.declared_sources;
        assert_eq!(
            connected_sources,
            declared_sources,
            "closure conversion generated an inconsistent closure-name boundary in `{theory_id}.{definition_name}` at edge e{edge_index}: operation `{operation}` is connected to {connected_sources} source wire(s), but its declaration expects {declared_sources}. connected source types: [{}]. declared source type map: `{}`.",
            objects_to_hexpr(&interface_types(definition, &edge.sources)),
            declaration.declaration_source_type_map,
        );
    }
}

fn declare_context_arrows_from_use_sites(
    syntax: &Theory,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition: &AnnotatedTerm,
    ambient_context_arity: usize,
) -> Result<(), ConvertTheoryError> {
    for (context_operation, context_use_site) in definition
        .hypergraph
        .edges
        .iter()
        .zip(&definition.hypergraph.adjacency)
        .filter(|(operation, _)| operation.as_str().starts_with(GENERATED_CONTEXT_PREFIX))
    {
        // The replacement graph is the source of truth for generated context
        // arrow boundaries: declare each operation from the concrete use-site
        // that closure conversion inserted into the converted definition.
        let raw_context_declaration = RawTheoryArrow {
            name: context_operation.clone(),
            type_maps: (
                boundary_objects_to_hexpr_in_context(
                    &interface_types(definition, &context_use_site.sources),
                    ambient_context_arity,
                ),
                boundary_objects_to_hexpr_in_context(
                    &interface_types(definition, &context_use_site.targets),
                    ambient_context_arity,
                ),
            ),
            definition: None,
        };
        let context_type_maps = interpret_type_maps(syntax, &raw_context_declaration.type_maps)?;
        arrows.insert(
            context_operation.clone(),
            TheoryArrow {
                name: context_operation.clone(),
                raw: raw_context_declaration,
                type_maps: context_type_maps,
                definition: None,
            },
        );
    }

    Ok(())
}

fn insert_closure_arrows(
    syntax: &Theory,
    theory_id: &TheoryId,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition_name: &Operation,
    closure: ConvertedClosure,
) -> Result<GeneratedClosureNameDeclaration, ConvertTheoryError> {
    let closure_name = closure.name(definition_name);
    let raw_closure = RawTheoryArrow {
        name: closure_name.clone(),
        type_maps: type_maps_for_term(&closure.term, closure.context.arity()),
        definition: Some(term_to_hexpr(&closure.term)),
    };
    let closure_type_maps = interpret_type_maps(syntax, &raw_closure.type_maps)?;
    arrows.insert(
        closure_name.clone(),
        TheoryArrow {
            raw: raw_closure.clone(),
            name: closure_name,
            type_maps: closure_type_maps,
            definition: Some(closure.term.map_nodes(|_| ())),
        },
    );

    let raw_name = name_symbols::name_arrow(syntax, &theory_id.0, &raw_closure)?;
    let name_type_maps = interpret_type_maps(syntax, &raw_name.type_maps)?;
    let name_boundary = GeneratedClosureNameDeclaration {
        operation: raw_name.name.clone(),
        declared_sources: name_type_maps.0.targets.len(),
        declaration_source_type_map: raw_name.type_maps.0.clone(),
    };
    arrows.insert(
        raw_name.name.clone(),
        TheoryArrow {
            name: raw_name.name.clone(),
            raw: raw_name,
            type_maps: name_type_maps,
            definition: None,
        },
    );

    Ok(name_boundary)
}

fn typed_definition(
    theory_id: &TheoryId,
    definition_name: &Operation,
    arrow: &TheoryArrow,
    theory_definition_types: Option<&BTreeMap<Operation, Vec<Obj>>>,
) -> Result<AnnotatedTerm, ConvertTheoryError> {
    let mut body =
        arrow
            .definition
            .clone()
            .ok_or_else(|| ConvertTheoryError::MissingDefinition {
                theory: theory_id.to_string(),
                definition: definition_name.to_string(),
            })?;
    body.quotient().ok();
    let labels = theory_definition_types
        .and_then(|types| types.get(definition_name))
        .cloned()
        .ok_or_else(|| ConvertTheoryError::MissingDefinitionTypes {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;
    body.with_nodes(|_| labels)
        .ok_or_else(|| ConvertTheoryError::NodeLabelCountMismatch {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })
}

fn primitive_closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
    let mut seen = BTreeSet::new();
    let mut wires = Vec::new();
    for (edge_index, operation) in definition.hypergraph.edges.iter().enumerate() {
        if converted_primitive(operation).is_none() {
            continue;
        }
        for &source in &definition.hypergraph.adjacency[edge_index].sources {
            if is_closure_type(&definition.hypergraph.nodes[source.0]) && seen.insert(source.0) {
                wires.push(source);
            }
        }
    }
    wires
}

fn rewrite_converted_primitives(definition: &mut AnnotatedTerm) {
    for operation in &mut definition.hypergraph.edges {
        if let Some(converted) = converted_primitive(operation) {
            *operation = op(converted);
        }
    }
}

fn converted_primitive(operation: &Operation) -> Option<&'static str> {
    CONVERTED_PRIMITIVES
        .iter()
        .find_map(|(source, target)| (operation.as_str() == *source).then_some(*target))
}

fn type_maps_for_term(term: &AnnotatedTerm, ambient_context_arity: usize) -> (Hexpr, Hexpr) {
    let source_types = interface_types(term, &term.sources);
    let target_types = interface_types(term, &term.targets);
    (
        objects_to_hexpr_in_context(&source_types, ambient_context_arity),
        objects_to_hexpr_in_context(&target_types, ambient_context_arity),
    )
}

fn contains_closure_type_map(type_map: &Term) -> bool {
    type_map
        .hypergraph
        .edges
        .iter()
        .any(|op| op.as_str() == FN_HOM_TYPE)
}

fn objects_to_hexpr_in_context(objects: &[Obj], ambient_context_arity: usize) -> Hexpr {
    let leaves = leaf_indices(objects);
    if let Some(max_leaf) = leaves.iter().max() {
        assert!(
            *max_leaf < ambient_context_arity,
            "object leaf index {max_leaf} is outside ambient context arity {ambient_context_arity}"
        );
    }
    let context_vars = context_vars(ambient_context_arity);
    let used_context_vars = leaves
        .into_iter()
        .map(|leaf| context_vars[leaf].clone())
        .collect();
    Hexpr::Composition(vec![
        Hexpr::Frobenius {
            sources: context_vars,
            targets: used_context_vars,
        },
        objects_to_hexpr(objects),
    ])
}

fn boundary_objects_to_hexpr_in_context(objects: &[Obj], ambient_context_arity: usize) -> Hexpr {
    if objects.is_empty() {
        return Hexpr::Frobenius {
            sources: context_vars(ambient_context_arity),
            targets: vec![],
        };
    }
    objects_to_hexpr_in_context(objects, ambient_context_arity)
}

fn leaf_indices(objects: &[Obj]) -> Vec<usize> {
    let mut indices = Vec::new();
    for object in objects {
        collect_leaf_indices(object, &mut indices);
    }
    indices
}

fn collect_leaf_indices(object: &Obj, indices: &mut Vec<usize>) {
    match object {
        Tree::Empty => {}
        Tree::Leaf(index, _) => indices.push(*index),
        Tree::Node(_, _, children) => {
            for child in children {
                collect_leaf_indices(child, indices);
            }
        }
    }
}

fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn interpret_type_maps(
    syntax: &Theory,
    type_maps: &(Hexpr, Hexpr),
) -> Result<(Term, Term), ConvertTheoryError> {
    let source = interpret_type_map(syntax, &type_maps.0)?;
    let target = interpret_type_map(syntax, &type_maps.1)?;
    if source.sources != target.sources {
        return Err(ConvertTheoryError::TypeMapDomainMismatch);
    }
    Ok((source, target))
}

fn interpret_type_map(syntax: &Theory, map: &Hexpr) -> Result<Term, ConvertTheoryError> {
    try_interpret(&syntax.local_signature(), map)
        .map(|term| term.map_nodes(|_| ()))
        .map_err(|error| ConvertTheoryError::TypeMapInterpretation {
            map: map.clone(),
            error,
        })
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == FN_HOM_TYPE && children.len() == 2
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

fn context_vars(arity: usize) -> Vec<Variable> {
    (0..arity).map(context_var).collect()
}

fn context_var(index: usize) -> Variable {
    format!("{GENERATED_VARIABLE_PREFIX}closure_ctx{index}")
        .parse()
        .expect("generated variable should parse")
}
