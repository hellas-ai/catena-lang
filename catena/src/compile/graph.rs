use std::collections::{HashMap, HashSet};

use hexpr::Operation;
use metacat::{
    check::check,
    theory::{Theory, TheoryId, TheorySet},
};
use open_hypergraphs::{
    category::Arrow,
    lax::{OpenHypergraph as LaxOpenHypergraph, functor::Functor},
    strict::vec::OpenHypergraph as StrictOpenHypergraph,
};
use thiserror::Error;

use crate::{compile::config::CompileConfig, pass::inline::Inline};

type DefinitionGraph = LaxOpenHypergraph<(), Operation>;
type LabeledGraph = LaxOpenHypergraph<String, Operation>;
type StrictLabeledGraph = StrictOpenHypergraph<String, Operation>;

#[derive(Clone, Debug)]
pub struct CompileGraph {
    pub theory: String,
    pub definition: String,
    pub graph: StrictLabeledGraph,
    pub children: Vec<NestedCompileGraph>,
}

#[derive(Clone, Debug)]
pub struct NestedCompileGraph {
    pub operation: String,
    pub graph: CompileGraph,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphCompileOptions {
    pub no_inline: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphCompileLimits {
    max_depth: usize,
    max_inline_iterations: usize,
}

impl Default for GraphCompileLimits {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_inline_iterations: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DefinitionRef {
    theory: String,
    definition: String,
}

impl DefinitionRef {
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
    limits: GraphCompileLimits,
    stack: Vec<DefinitionRef>,
}

#[derive(Error, Debug)]
pub enum CompileGraphError {
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

pub fn compile_graph(
    set: &TheorySet,
    config: &CompileConfig,
    theory: &str,
    definition: &str,
) -> Result<CompileGraph, CompileGraphError> {
    compile_graph_with_options(
        set,
        config,
        theory,
        definition,
        GraphCompileOptions::default(),
    )
}

pub fn compile_graph_with_options(
    set: &TheorySet,
    config: &CompileConfig,
    theory: &str,
    definition: &str,
    options: GraphCompileOptions,
) -> Result<CompileGraph, CompileGraphError> {
    let mut state = GraphCompileState {
        set,
        config,
        options,
        limits: GraphCompileLimits::default(),
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
        if self.stack.len() > self.limits.max_depth {
            return Err(CompileGraphError::NestedLimit(format!(
                "{theory_name}.{definition}"
            )));
        }

        let current = DefinitionRef::new(theory_name, definition);
        if let Some(index) = self.stack.iter().position(|entry| entry == &current) {
            // For now graph rendering rejects cyclic cross-theory definitions.
            // We may relax this later and render recursive definitions with
            // back-references instead of expanding them.
            let mut cycle = self.stack[index..]
                .iter()
                .map(DefinitionRef::label)
                .collect::<Vec<_>>();
            cycle.push(current.label());
            return Err(CompileGraphError::NestedCycle { cycle });
        }

        self.stack.push(current);
        let result = self.compile_graph_pipeline(theory_name, definition);
        self.stack.pop();
        result
    }

    fn compile_graph_pipeline(
        &mut self,
        theory_name: &str,
        definition: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        let syntax = self.syntax_theory()?;
        let theory = self.theory(theory_name)?;
        let definition_key = parse_operation(definition)?;
        let graph = self.compile_definition_graph(theory_name, theory, syntax, &definition_key)?;
        let children = self.compile_nested_foreign_graphs(theory_name, &graph)?;

        Ok(CompileGraph {
            theory: theory_name.to_string(),
            definition: definition.to_string(),
            graph,
            children,
        })
    }

    fn syntax_theory(&self) -> Result<&Theory, CompileGraphError> {
        self.theory(self.config.syntax)
    }

    fn theory(&self, theory_name: &str) -> Result<&Theory, CompileGraphError> {
        let id = TheoryId(
            theory_name
                .parse()
                .map_err(|_| CompileGraphError::UnknownTheory(theory_name.to_string()))?,
        );
        self.set
            .theories
            .get(&id)
            .ok_or_else(|| CompileGraphError::UnknownTheory(theory_name.to_string()))
    }

    fn compile_definition_graph(
        &self,
        theory_name: &str,
        theory: &Theory,
        syntax: &Theory,
        definition_key: &Operation,
    ) -> Result<StrictLabeledGraph, CompileGraphError> {
        let arrow = theory
            .get_arrow(definition_key)
            .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
        let mut graph = arrow
            .definition
            .clone()
            .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
        let definitions = inline_definitions(theory, theory_name, &self.options.no_inline)?;
        graph = inline_local_definitions(
            graph,
            &definitions,
            self.limits.max_inline_iterations,
            definition_key,
        )?;

        let graph = typecheck_and_label_graph(
            theory,
            syntax,
            definition_key.as_str(),
            arrow.type_maps.0.clone(),
            arrow.type_maps.1.clone(),
            graph,
        )?;

        Ok(graph.to_strict())
    }

    fn compile_nested_foreign_graphs(
        &mut self,
        theory_name: &str,
        graph: &StrictLabeledGraph,
    ) -> Result<Vec<NestedCompileGraph>, CompileGraphError> {
        let mut seen = HashSet::new();
        let mut children = Vec::new();

        for operation in graph.h.x.0.iter() {
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

            let graph = self.compile_foreign_child(extension.source, local_name)?;
            children.push(NestedCompileGraph {
                operation: operation_name,
                graph,
            });
        }

        Ok(children)
    }

    fn compile_foreign_child(
        &mut self,
        source_theory: &str,
        local_name: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        let native_foreign_theory = self.theory(source_theory)?;
        let fully_qualified = format!("{source_theory}.{local_name}");
        if definition_exists(native_foreign_theory, local_name)?
            && !matches_any_no_inline(&fully_qualified, &self.options.no_inline)
        {
            self.compile_nested_graph(source_theory, local_name)
        } else {
            let syntax = self.syntax_theory()?;
            compile_primitive_child_graph(syntax, native_foreign_theory, source_theory, local_name)
        }
    }
}

fn inline_local_definitions(
    mut graph: DefinitionGraph,
    definitions: &HashMap<Operation, DefinitionGraph>,
    max_inline_iterations: usize,
    definition_key: &Operation,
) -> Result<DefinitionGraph, CompileGraphError> {
    for _ in 0..max_inline_iterations {
        let inlinable = inlinable_edges(&graph, definitions);
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

fn typecheck_and_label_graph(
    theory: &Theory,
    syntax: &Theory,
    definition: &str,
    source: DefinitionGraph,
    target: DefinitionGraph,
    mut graph: DefinitionGraph,
) -> Result<LabeledGraph, CompileGraphError> {
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

fn compile_primitive_child_graph(
    syntax: &Theory,
    theory: &Theory,
    theory_name: &str,
    local_name: &str,
) -> Result<CompileGraph, CompileGraphError> {
    let operation = parse_operation(local_name)?;
    let arrow = theory
        .get_arrow(&operation)
        .ok_or_else(|| CompileGraphError::UnknownOperation(local_name.to_string()))?;
    let graph = LaxOpenHypergraph::singleton(
        operation,
        vec![(); arrow.type_maps.0.target().len()],
        vec![(); arrow.type_maps.1.target().len()],
    );
    let graph = typecheck_and_label_graph(
        theory,
        syntax,
        local_name,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        graph,
    )?
    .to_strict();

    Ok(CompileGraph {
        theory: theory_name.to_string(),
        definition: local_name.to_string(),
        graph,
        children: Vec::new(),
    })
}

fn inline_definitions(
    theory: &Theory,
    theory_name: &str,
    no_inline: &[String],
) -> Result<HashMap<Operation, DefinitionGraph>, CompileGraphError> {
    let Theory::Theory { arrows, .. } = theory else {
        return Err(CompileGraphError::NotUserTheory("nat".to_string()));
    };
    Ok(arrows
        .iter()
        .filter_map(|(name, arrow)| {
            let local_name = name.to_string();
            let fully_qualified = format!("{theory_name}.{local_name}");
            if matches_any_no_inline(&fully_qualified, no_inline)
                || matches_any_no_inline(&local_name, no_inline)
            {
                None
            } else {
                arrow.definition.clone().map(|term| (name.clone(), term))
            }
        })
        .collect())
}

fn inlinable_edges(
    graph: &DefinitionGraph,
    definitions: &HashMap<Operation, DefinitionGraph>,
) -> HashSet<Operation> {
    graph
        .hypergraph
        .edges
        .iter()
        .filter(|operation| definitions.contains_key(*operation))
        .cloned()
        .collect()
}

fn parse_operation(name: &str) -> Result<Operation, CompileGraphError> {
    name.parse()
        .map_err(|_| CompileGraphError::InvalidDefinition(name.to_string()))
}

fn matches_any_no_inline(name: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| glob_match(pattern, name))
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.as_bytes();
    let text = text.as_bytes();
    let (mut p, mut t) = (0, 0);
    let mut star = None;
    let mut after_star_text = 0;

    while t < text.len() {
        if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            after_star_text = t;
        } else if p < pattern.len() && pattern[p] == text[t] {
            p += 1;
            t += 1;
        } else if let Some(star_index) = star {
            p = star_index + 1;
            after_star_text += 1;
            t = after_star_text;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }

    p == pattern.len()
}
