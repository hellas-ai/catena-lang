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
        config::CompileConfig,
        lift::{LiftError, lift_with_tensor},
    },
    pass::inline::Inline,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CompileGraph {
    pub theory: String,
    pub definition: String,
    pub graph: OpenHypergraph<String, Operation>,
    pub children: Vec<NestedCompileGraph>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NestedCompileGraph {
    pub operation: String,
    pub graph: CompileGraph,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphCompileOptions {
    max_depth: usize,
    max_inline_iterations: usize,
}

impl Default for GraphCompileOptions {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_inline_iterations: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GraphDefinition {
    theory: String,
    definition: String,
}

impl GraphDefinition {
    fn new(theory: &str, definition: &str) -> Self {
        Self {
            theory: theory.to_string(),
            definition: definition.to_string(),
        }
    }

    fn label(&self) -> String {
        format!("{}.{}", self.theory, self.definition)
    }
}

struct GraphCompileState<'a> {
    set: &'a TheorySet,
    config: &'a CompileConfig,
    options: GraphCompileOptions,
    stack: Vec<GraphDefinition>,
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
    #[error("too-deep nested graph expansion while rendering `{0}`")]
    NestedLimit(String),
    #[error("cyclic nested graph expansion: {}", .cycle.join(" -> "))]
    NestedCycle { cycle: Vec<String> },
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
    config: &CompileConfig,
    theory: &str,
    definition: &str,
) -> Result<CompileGraph, CompileGraphError> {
    let mut state = GraphCompileState {
        set,
        config,
        options: GraphCompileOptions::default(),
        stack: Vec::new(),
    };
    state.compile_nested_graph(theory, definition)
}

impl GraphCompileState<'_> {
    fn compile_nested_graph(
        &mut self,
        theory_name: &str,
        definition: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        if self.stack.len() > self.options.max_depth {
            return Err(CompileGraphError::NestedLimit(format!(
                "{theory_name}.{definition}"
            )));
        }

        let current = GraphDefinition::new(theory_name, definition);
        if let Some(index) = self.stack.iter().position(|entry| entry == &current) {
            // For now graph rendering rejects cyclic cross-theory definitions.
            // We may relax this later and render recursive definitions with
            // back-references instead of expanding them.
            let mut cycle = self.stack[index..]
                .iter()
                .map(GraphDefinition::label)
                .collect::<Vec<_>>();
            cycle.push(current.label());
            return Err(CompileGraphError::NestedCycle { cycle });
        }

        self.stack.push(current);
        let result = self.compile_nested_graph_inner(theory_name, definition);
        self.stack.pop();
        result
    }

    fn compile_nested_graph_inner(
        &mut self,
        theory_name: &str,
        definition: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        let syntax = lookup_theory(self.set, self.config.syntax)?;
        let bundle = graph_theory(self.set, self.config, syntax, theory_name)?;
        let definition_key = parse_operation(definition)?;
        let graph = strictify(annotated_definition_graph(
            &bundle,
            syntax,
            &definition_key,
            self.options.max_inline_iterations,
        )?);
        let children = self.nested_graphs(theory_name, &graph)?;

        Ok(CompileGraph {
            theory: theory_name.to_string(),
            definition: definition.to_string(),
            graph,
            children,
        })
    }

    fn nested_graphs(
        &mut self,
        theory_name: &str,
        graph: &OpenHypergraph<String, Operation>,
    ) -> Result<Vec<NestedCompileGraph>, CompileGraphError> {
        let mut seen = HashSet::new();
        let mut children = Vec::new();

        for operation in &graph.hypergraph.edges {
            let operation_name = operation.to_string();
            let Some((foreign_theory_name, local_name)) = operation_name.split_once('.') else {
                continue;
            };
            let Some(extension) = self
                .config
                .extension_for_target_and_prefix(theory_name, foreign_theory_name)
            else {
                continue;
            };
            if !seen.insert(operation_name.clone()) {
                continue;
            }

            let native_foreign_theory = lookup_theory(self.set, extension.source)?;
            let graph = if definition_exists(native_foreign_theory, local_name)? {
                self.compile_nested_graph(extension.source, local_name)?
            } else {
                let syntax = lookup_theory(self.set, self.config.syntax)?;
                primitive_graph(syntax, native_foreign_theory, extension.source, local_name)?
            };
            children.push(NestedCompileGraph {
                operation: operation_name,
                graph,
            });
        }

        Ok(children)
    }
}

fn graph_theory(
    set: &TheorySet,
    config: &CompileConfig,
    syntax: &Theory,
    theory_name: &str,
) -> Result<Theory, CompileGraphError> {
    let mut theory = lookup_theory(set, theory_name)?.clone();
    let excluded_prefixes = config.lifted_prefixes();

    for extension in config.extensions_for_target(theory_name) {
        let source = lookup_theory(set, extension.source)?;
        theory = lift_with_tensor(
            source,
            &theory,
            syntax,
            extension.prefix,
            extension.tensor,
            extension.unit,
            &excluded_prefixes,
        )?;
    }

    Ok(theory)
}

fn locally_inlined_definition(
    theory: &Theory,
    definition_key: &Operation,
    max_inline_iterations: usize,
) -> Result<OpenHypergraph<(), Operation>, CompileGraphError> {
    let mut graph = definition_term(theory, definition_key)?;
    let definitions = inline_definitions(theory)?;

    for _ in 0..max_inline_iterations {
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
    max_inline_iterations: usize,
) -> Result<OpenHypergraph<String, Operation>, CompileGraphError> {
    let arrow = theory
        .get_arrow(definition_key)
        .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
    let graph = locally_inlined_definition(theory, definition_key, max_inline_iterations)?;
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
    theory_name: &str,
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
        theory: theory_name.to_string(),
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
