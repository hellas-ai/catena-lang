use std::collections::{BTreeMap, BTreeSet};

use hexpr::{Hexpr, Operation, interpret::Error as HexprInterpretError, try_interpret};
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
};

const IF_OPS: &[&str] = &["if", "bool.if"];

type Obj = Tree<(), Operation>;

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

    let mut converted_arrows = arrows.clone();
    for (definition_name, arrow) in arrows {
        if arrow.definition.is_none() {
            continue;
        }

        let typed = typed_definition(theory_id, definition_name, arrow, theory_definition_types)?;
        let closure_wires = if_closure_wires(&typed);
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

fn update_definition_arrow(
    syntax: &Theory,
    theory_id: &TheoryId,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition_name: &Operation,
    original: &TheoryArrow,
    converted: Converted,
) -> Result<(), ConvertTheoryError> {
    let mut converted_definition = converted.definition;
    rewrite_if_operations(&mut converted_definition);

    let mut raw = original.raw.clone();
    raw.definition = Some(term_to_hexpr(&converted_definition));
    let mut arrow = original.clone();
    arrow.raw = raw;
    arrow.definition = Some(converted_definition.map_nodes(|_| ()));
    arrows.insert(definition_name.clone(), arrow);

    for closure in converted.closures {
        insert_closure_arrows(syntax, theory_id, arrows, definition_name, closure)?;
    }

    Ok(())
}

fn insert_closure_arrows(
    syntax: &Theory,
    theory_id: &TheoryId,
    arrows: &mut BTreeMap<Operation, TheoryArrow>,
    definition_name: &Operation,
    closure: ConvertedClosure,
) -> Result<(), ConvertTheoryError> {
    let closure_name = closure.name(definition_name);
    let raw_closure = RawTheoryArrow {
        name: closure_name.clone(),
        type_maps: type_maps_for_term(&closure.term),
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
    arrows.insert(
        raw_name.name.clone(),
        TheoryArrow {
            name: raw_name.name.clone(),
            raw: raw_name,
            type_maps: name_type_maps,
            definition: None,
        },
    );

    Ok(())
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

fn if_closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
    let mut seen = BTreeSet::new();
    let mut wires = Vec::new();
    for (edge_index, operation) in definition.hypergraph.edges.iter().enumerate() {
        if !IF_OPS.contains(&operation.as_str()) {
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

fn rewrite_if_operations(definition: &mut AnnotatedTerm) {
    for operation in &mut definition.hypergraph.edges {
        match operation.as_str() {
            "if" => *operation = op("ifc"),
            "bool.if" => *operation = op("bool.ifc"),
            _ => {}
        }
    }
}

fn type_maps_for_term(term: &AnnotatedTerm) -> (Hexpr, Hexpr) {
    (
        objects_to_hexpr(&interface_types(term, &term.sources)),
        objects_to_hexpr(&interface_types(term, &term.targets)),
    )
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
    operation.as_str() == "=>" && children.len() == 2
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
