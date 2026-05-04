use std::collections::{HashMap, HashSet};

use hexpr::{Operation, try_interpret};
use metacat::{
    check::check,
    syntax::{Declaration, TheoryBundle},
    theory::OperationKey,
};
use open_hypergraphs::{
    category::Arrow,
    lax::{OpenHypergraph, functor::Functor},
};
use thiserror::Error;

use crate::{
    compile::lift::{LiftError, lift_control_to_data, lift_data_to_control},
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
    pub graph: OpenHypergraph<String, OperationKey>,
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
    #[error("invalid definition name `{0}`")]
    InvalidDefinition(String),
    #[error("unknown definition `{0}`")]
    UnknownDefinition(String),
    #[error("unknown operation `{0}`")]
    UnknownOperation(String),
    #[error("invalid hexpr: {0}")]
    InvalidHexpr(#[from] hexpr::interpret::Error<metacat::theory::Error>),
    #[error("definition {definition} failed typecheck: {error:?}")]
    Typecheck {
        definition: String,
        error: metacat::check::Error<OperationKey>,
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

pub fn compile_graph(
    data: &TheoryBundle,
    control: &TheoryBundle,
    theory: GraphTheory,
    definition: &str,
) -> Result<CompileGraph, CompileGraphError> {
    compile_graph_at_depth(data, control, theory, definition, 0)
}

fn compile_graph_at_depth(
    data: &TheoryBundle,
    control: &TheoryBundle,
    theory: GraphTheory,
    definition: &str,
    depth: usize,
) -> Result<CompileGraph, CompileGraphError> {
    if depth > 32 {
        return Err(CompileGraphError::NestedLimit(definition.to_string()));
    }

    let bundle = match theory {
        GraphTheory::Data => lift_control_to_data(control, data)?,
        GraphTheory::Control => lift_data_to_control(data, control)?,
    };
    let foreign_bundle = match theory {
        GraphTheory::Data => control,
        GraphTheory::Control => data,
    };
    let definition_key = parse_operation(definition)?;
    let declaration = bundle
        .definitions
        .get(&definition_key)
        .ok_or_else(|| CompileGraphError::UnknownDefinition(definition.to_string()))?;
    let graph = strictify(annotated_definition_graph(
        &bundle,
        &definition_key,
        declaration,
    )?);
    let children = nested_graphs(data, control, theory, foreign_bundle, &graph, depth)?;

    Ok(CompileGraph {
        theory,
        definition: definition.to_string(),
        graph,
        children,
    })
}

fn locally_inlined_definition(
    bundle: &TheoryBundle,
    definition_key: &Operation,
) -> Result<OpenHypergraph<(), OperationKey>, CompileGraphError> {
    let mut graph = definition_term(bundle, definition_key)?;
    let definitions = inline_definitions(bundle)?;

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
    bundle: &TheoryBundle,
    definition_key: &Operation,
    declaration: &Declaration,
) -> Result<OpenHypergraph<String, OperationKey>, CompileGraphError> {
    let graph = locally_inlined_definition(bundle, definition_key)?;
    let source = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.source_map,
    )?);
    let target = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.target_map,
    )?);
    annotated_graph(bundle, definition_key.as_str(), source, target, graph)
}

fn annotated_graph(
    bundle: &TheoryBundle,
    definition: &str,
    source: OpenHypergraph<(), OperationKey>,
    target: OpenHypergraph<(), OperationKey>,
    mut graph: OpenHypergraph<(), OperationKey>,
) -> Result<OpenHypergraph<String, OperationKey>, CompileGraphError> {
    let labels = check(&bundle.arrow_theory, source, target, &mut graph)
        .map_err(|error| CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error,
        })?
        .into_iter()
        .map(|tree| {
            tree.pretty(Some(&|op| {
                bundle.object_theory.type_maps(op).1.targets.len()
            }))
        })
        .collect::<Vec<_>>();

    graph
        .with_nodes(|_| labels)
        .ok_or_else(|| CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error: metacat::check::Error::InvalidTypeMaps,
        })
}

fn nested_graphs(
    data: &TheoryBundle,
    control: &TheoryBundle,
    theory: GraphTheory,
    native_bundle: &TheoryBundle,
    graph: &OpenHypergraph<String, OperationKey>,
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

        let graph = if foreign_definition_exists(native_bundle, local_name)? {
            compile_graph_at_depth(
                data,
                control,
                theory.foreign_theory(),
                local_name,
                depth + 1,
            )?
        } else {
            primitive_graph(native_bundle, theory.foreign_theory(), local_name)?
        };
        children.push(NestedCompileGraph {
            operation: operation_name,
            graph,
        });
    }

    Ok(children)
}

fn foreign_definition_exists(
    bundle: &TheoryBundle,
    local_name: &str,
) -> Result<bool, CompileGraphError> {
    Ok(bundle
        .definitions
        .contains_key(&parse_operation(local_name)?))
}

fn primitive_graph(
    bundle: &TheoryBundle,
    theory: GraphTheory,
    local_name: &str,
) -> Result<CompileGraph, CompileGraphError> {
    let operation = bundle
        .arrow_theory
        .get_operation_key(local_name)
        .ok_or_else(|| CompileGraphError::UnknownOperation(local_name.to_string()))?;
    let (source, target) = bundle.arrow_theory.type_maps(&operation);
    let graph = OpenHypergraph::singleton(
        operation,
        vec![(); source.target().len()],
        vec![(); target.target().len()],
    );
    let graph = strictify(annotated_graph(
        bundle,
        local_name,
        source.clone(),
        target.clone(),
        graph,
    )?);

    Ok(CompileGraph {
        theory,
        definition: local_name.to_string(),
        graph,
        children: Vec::new(),
    })
}

fn inline_definitions(
    bundle: &TheoryBundle,
) -> Result<HashMap<OperationKey, OpenHypergraph<(), OperationKey>>, CompileGraphError> {
    let mut definitions = HashMap::new();
    for name in bundle.definitions.keys() {
        let key = bundle
            .arrow_theory
            .get_operation_key(name.as_str())
            .ok_or_else(|| CompileGraphError::UnknownOperation(name.to_string()))?;
        definitions.insert(key, definition_term(bundle, name)?);
    }
    Ok(definitions)
}

fn inlinable_edges(
    graph: &OpenHypergraph<(), OperationKey>,
    definitions: &HashMap<OperationKey, OpenHypergraph<(), OperationKey>>,
) -> HashSet<OperationKey> {
    graph
        .hypergraph
        .edges
        .iter()
        .filter(|operation| definitions.contains_key(*operation))
        .cloned()
        .collect()
}

fn definition_term(
    bundle: &TheoryBundle,
    key: &Operation,
) -> Result<OpenHypergraph<(), OperationKey>, CompileGraphError> {
    let declaration = bundle
        .definitions
        .get(key)
        .ok_or_else(|| CompileGraphError::UnknownDefinition(key.to_string()))?;
    declaration_term(bundle, declaration)
}

fn declaration_term(
    bundle: &TheoryBundle,
    declaration: &Declaration,
) -> Result<OpenHypergraph<(), OperationKey>, CompileGraphError> {
    let hexpr = declaration
        .definition
        .as_ref()
        .expect("definition entries always have a body");
    Ok(forget_labels(try_interpret(&bundle.arrow_theory, hexpr)?))
}

fn parse_operation(name: &str) -> Result<Operation, CompileGraphError> {
    name.parse()
        .map_err(|_| CompileGraphError::InvalidDefinition(name.to_string()))
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}

fn strictify<O: Clone + PartialEq>(
    graph: OpenHypergraph<O, OperationKey>,
) -> OpenHypergraph<O, OperationKey> {
    OpenHypergraph::from_strict(graph.to_strict())
}
