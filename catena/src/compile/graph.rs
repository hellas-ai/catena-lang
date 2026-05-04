use std::collections::{HashMap, HashSet};

use hexpr::Operation;
use metacat::{
    check::check,
    theory::{Theory, TheorySet},
};
use open_hypergraphs::{
    category::Arrow,
    lax::{OpenHypergraph, functor::Functor},
};
use thiserror::Error;

use crate::{
    compile::{
        check::{CheckError, theory as lookup_theory},
        lift::{LiftError, lift_control_to_data, lift_data_to_control},
    },
    pass::inline::Inline,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphTheory {
    Data,
    Control,
}

impl GraphTheory {
    pub fn foreign_prefix(self) -> &'static str {
        match self {
            Self::Data => "control",
            Self::Control => "data",
        }
    }

    pub fn foreign_theory(self) -> Self {
        match self {
            Self::Data => Self::Control,
            Self::Control => Self::Data,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileGraph {
    pub theory: GraphTheory,
    pub definition: String,
    pub graph: OpenHypergraph<String, Operation>,
    pub children: Vec<NestedCompileGraph>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NestedCompileGraph {
    pub operation: String,
    pub graph: CompileGraph,
}

#[derive(Error, Debug)]
pub enum CompileGraphError {
    #[error("{error}")]
    Lift { error: LiftError },
    #[error("unknown theory `{0}`")]
    UnknownTheory(String),
    #[error("theory `{0}` is not a user theory")]
    NotUserTheory(String),
    #[error("invalid definition name `{0}`")]
    InvalidDefinition(String),
    #[error("unknown definition `{0}`")]
    UnknownDefinition(String),
    #[error("unknown operation `{0}`")]
    UnknownOperation(String),
    #[error("definition {definition} failed typecheck: {error:?}")]
    Typecheck {
        definition: String,
        error: metacat::check::Error<Operation>,
    },
    #[error("recursive or too-deep inline expansion while rendering `{0}`")]
    InlineLimit(String),
    #[error("recursive or too-deep nested graph expansion while rendering `{0}`")]
    NestedLimit(String),
}

impl From<LiftError> for CompileGraphError {
    fn from(error: LiftError) -> Self {
        Self::Lift { error }
    }
}

impl From<CheckError> for CompileGraphError {
    fn from(error: CheckError) -> Self {
        match error {
            CheckError::UnknownTheory(name) => Self::UnknownTheory(name),
            CheckError::NotUserTheory(name) => Self::NotUserTheory(name),
            CheckError::Lift { error } => Self::Lift { error },
            CheckError::Typecheck { definition, error } => Self::Typecheck { definition, error },
        }
    }
}

pub fn compile_graph(
    set: &TheorySet,
    theory: GraphTheory,
    definition: &str,
) -> Result<CompileGraph, CompileGraphError> {
    compile_graph_at_depth(set, theory, definition, 0)
}

fn compile_graph_at_depth(
    set: &TheorySet,
    theory: GraphTheory,
    definition: &str,
    depth: usize,
) -> Result<CompileGraph, CompileGraphError> {
    if depth > 32 {
        return Err(CompileGraphError::NestedLimit(definition.to_string()));
    }

    let syntax = lookup_theory(set, "syntax")?;
    let data = lookup_theory(set, "data")?;
    let control = lookup_theory(set, "control")?;
    let bundle = match theory {
        GraphTheory::Data => lift_control_to_data(control, data, syntax)?,
        GraphTheory::Control => lift_data_to_control(data, control, syntax)?,
    };
    let foreign_theory = match theory {
        GraphTheory::Data => control,
        GraphTheory::Control => data,
    };
    let definition_key = parse_operation(definition)?;
    let graph = strictify(annotated_definition_graph(
        &bundle,
        syntax,
        &definition_key,
    )?);
    let children = nested_graphs(set, theory, foreign_theory, &graph, depth)?;

    Ok(CompileGraph {
        theory,
        definition: definition.to_string(),
        graph,
        children,
    })
}

fn locally_inlined_definition(
    theory: &Theory,
    definition_key: &Operation,
) -> Result<OpenHypergraph<(), Operation>, CompileGraphError> {
    let mut graph = definition_term(theory, definition_key)?;
    let definitions = inline_definitions(theory)?;

    for _ in 0..64 {
        let inlinable = inlinable_edges(&graph, &definitions);
        if inlinable.is_empty() {
            return Ok(graph);
        }

        graph.quotient().expect("quotient should be defined");
        graph = Inline {
            definitions: definitions.clone(),
        }
        .map_arrow(&graph);
    }

    Err(CompileGraphError::InlineLimit(definition_key.to_string()))
}

fn annotated_definition_graph(
    theory: &Theory,
    syntax: &Theory,
    definition_key: &Operation,
) -> Result<OpenHypergraph<String, Operation>, CompileGraphError> {
    let arrow = theory
        .get_arrow(definition_key)
        .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
    let graph = locally_inlined_definition(theory, definition_key)?;
    annotated_graph(
        theory,
        syntax,
        definition_key.as_str(),
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        graph,
    )
}

fn annotated_graph(
    theory: &Theory,
    syntax: &Theory,
    definition: &str,
    source: OpenHypergraph<(), Operation>,
    target: OpenHypergraph<(), Operation>,
    mut graph: OpenHypergraph<(), Operation>,
) -> Result<OpenHypergraph<String, Operation>, CompileGraphError> {
    let labels = check(theory, source, target, &mut graph)
        .map_err(|error| CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error,
        })?
        .into_iter()
        .map(|tree| {
            tree.try_pretty(Some(&|op| {
                syntax
                    .coarity_of(op)
                    .ok_or_else(|| CompileGraphError::UnknownOperation(op.to_string()))
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;

    graph
        .with_nodes(|_| labels)
        .ok_or_else(|| CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error: metacat::check::Error::InvalidTypeMaps,
        })
}

fn nested_graphs(
    set: &TheorySet,
    theory: GraphTheory,
    native_foreign_theory: &Theory,
    graph: &OpenHypergraph<String, Operation>,
    depth: usize,
) -> Result<Vec<NestedCompileGraph>, CompileGraphError> {
    let foreign_prefix = theory.foreign_prefix();
    let mut seen = HashSet::new();
    let mut children = Vec::new();

    for operation in &graph.hypergraph.edges {
        let operation_name = operation.to_string();
        let Some(local_name) = operation_name.strip_prefix(&format!("{foreign_prefix}.")) else {
            continue;
        };
        if !seen.insert(operation_name.clone()) {
            continue;
        }

        let graph = if definition_exists(native_foreign_theory, local_name)? {
            compile_graph_at_depth(set, theory.foreign_theory(), local_name, depth + 1)?
        } else {
            let syntax = lookup_theory(set, "syntax")?;
            primitive_graph(
                syntax,
                native_foreign_theory,
                theory.foreign_theory(),
                local_name,
            )?
        };
        children.push(NestedCompileGraph {
            operation: operation_name,
            graph,
        });
    }

    Ok(children)
}

fn definition_exists(theory: &Theory, local_name: &str) -> Result<bool, CompileGraphError> {
    let operation = parse_operation(local_name)?;
    Ok(theory
        .get_arrow(&operation)
        .and_then(|arrow| arrow.definition.as_ref())
        .is_some())
}

fn primitive_graph(
    syntax: &Theory,
    theory: &Theory,
    graph_theory: GraphTheory,
    local_name: &str,
) -> Result<CompileGraph, CompileGraphError> {
    let operation = parse_operation(local_name)?;
    let arrow = theory
        .get_arrow(&operation)
        .ok_or_else(|| CompileGraphError::UnknownOperation(local_name.to_string()))?;
    let graph = OpenHypergraph::singleton(
        operation,
        vec![(); arrow.type_maps.0.target().len()],
        vec![(); arrow.type_maps.1.target().len()],
    );
    let graph = strictify(annotated_graph(
        theory,
        syntax,
        local_name,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        graph,
    )?);

    Ok(CompileGraph {
        theory: graph_theory,
        definition: local_name.to_string(),
        graph,
        children: Vec::new(),
    })
}

fn inline_definitions(
    theory: &Theory,
) -> Result<HashMap<Operation, OpenHypergraph<(), Operation>>, CompileGraphError> {
    let Theory::Theory { arrows, .. } = theory else {
        return Err(CompileGraphError::NotUserTheory("nat".to_string()));
    };
    Ok(arrows
        .iter()
        .filter_map(|(name, arrow)| arrow.definition.clone().map(|term| (name.clone(), term)))
        .collect())
}

fn inlinable_edges(
    graph: &OpenHypergraph<(), Operation>,
    definitions: &HashMap<Operation, OpenHypergraph<(), Operation>>,
) -> HashSet<Operation> {
    graph
        .hypergraph
        .edges
        .iter()
        .filter(|operation| definitions.contains_key(*operation))
        .cloned()
        .collect()
}

fn definition_term(
    theory: &Theory,
    key: &Operation,
) -> Result<OpenHypergraph<(), Operation>, CompileGraphError> {
    theory
        .get_arrow(key)
        .and_then(|arrow| arrow.definition.clone())
        .ok_or_else(|| CompileGraphError::UnknownDefinition(key.to_string()))
}

fn parse_operation(name: &str) -> Result<Operation, CompileGraphError> {
    name.parse()
        .map_err(|_| CompileGraphError::InvalidDefinition(name.to_string()))
}

fn strictify<O: Clone + PartialEq>(
    graph: OpenHypergraph<O, Operation>,
) -> OpenHypergraph<O, Operation> {
    OpenHypergraph::from_strict(graph.to_strict())
}
