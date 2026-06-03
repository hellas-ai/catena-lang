use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use hexpr::{Hexpr, Operation, Variable, interpret::Signature};
use metacat::{
    check::check,
    theory::{Theory, TheoryId, TheorySet},
};
use open_hypergraphs::{
    category::Arrow,
    lax::{Interface, NodeId, OpenHypergraph as LaxOpenHypergraph, functor::Functor},
    strict::vec::OpenHypergraph as StrictOpenHypergraph,
};
use thiserror::Error;

use crate::{compile::config::CompileConfig, lang::Obj, pass::inline::Inline};

type DefinitionGraph = LaxOpenHypergraph<(), Operation>;
type TypedGraph = LaxOpenHypergraph<Obj, Operation>;
type StrictTypedGraph = StrictOpenHypergraph<Obj, Operation>;

// CompileGraph construction has four responsibilities:
//
// 1. Load one checked definition as a hypergraph.
// 2. Inline only the local structural wrappers listed below.
// 3. Typecheck the resulting hypergraph to label every wire with semantic
//    object/type information while preserving source variable names for
//    diagnostics.
// 4. Discover child regions. Local definitions and valid cross-theory calls are
//    represented as nested CompileGraphs; primitive operations stay as edges.
//
// The important invariant is that theory boundaries are visible in the graph.
// Data/control tensor have different meanings, so cross-theory calls must be
// child regions unless they are true backend primitives.

// Local definitions are region calls by default. This list is deliberately
// small: only definitions that are pure structural convenience wrappers should
// be flattened during CompileGraph construction. Definitions that carry
// meaningful control/data boundaries, such as `if`, must remain as child
// regions so structured lowering can see them.
const INLINE_LOCAL_DEFINITIONS: &[&str] = &["data.if"];

#[derive(Clone, Debug)]
pub struct CompileGraph {
    pub theory: CompileTheory,
    pub definition_name: String,
    pub graph: StrictTypedGraph,
    pub source_variable_names: HashMap<usize, String>,
    pub children: Vec<NestedCompileGraph>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompileTheory {
    Data,
    Control,
}

impl CompileTheory {
    fn parse(name: &str) -> Result<Self, CompileGraphError> {
        match name {
            "data" => Ok(Self::Data),
            "control" => Ok(Self::Control),
            other => Err(CompileGraphError::UnknownTheory(other.to_string())),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Data => "data",
            Self::Control => "control",
        }
    }
}

impl fmt::Display for CompileTheory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct NestedCompileGraph {
    pub operation: String,
    pub graph: CompileGraph,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphBuildLimits {
    max_depth: usize,
    max_inline_iterations: usize,
}

impl Default for GraphBuildLimits {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_inline_iterations: 64,
        }
    }
}

struct CompiledBody {
    graph: StrictTypedGraph,
    source_variable_names: HashMap<usize, String>,
}

struct LoadedTheoryDefinition {
    source_type_map: DefinitionGraph,
    target_type_map: DefinitionGraph,
    graph: DefinitionGraph,
    source_variable_names: HashMap<usize, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DefinitionRef {
    theory: CompileTheory,
    definition: String,
}

impl DefinitionRef {
    fn new(theory: CompileTheory, definition: &str) -> Self {
        Self {
            theory,
            definition: definition.to_string(),
        }
    }

    fn label(&self) -> String {
        format!("{}.{}", self.theory, self.definition)
    }
}

struct CompileGraphBuilder<'a> {
    set: &'a TheorySet,
    config: &'a CompileConfig,
    inline_policy: InlinePolicy,
    limits: GraphBuildLimits,
    stack: Vec<DefinitionRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OperationRole {
    Primitive,
    LocalDefinition,
    CrossTheoryDefinition {
        source_theory: CompileTheory,
        local_name: String,
    },
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
    #[error("recursive or too-deep inline expansion while building `{0}`")]
    InlineLimit(String),
    #[error("too-deep nested graph expansion while building `{0}`")]
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
    let theory = CompileTheory::parse(theory)?;
    let mut builder = CompileGraphBuilder {
        set,
        config,
        inline_policy: InlinePolicy::default(),
        limits: GraphBuildLimits::default(),
        stack: Vec::new(),
    };
    builder.compile_region(theory, definition)
}

impl CompileGraphBuilder<'_> {
    fn compile_region(
        &mut self,
        compile_theory: CompileTheory,
        definition: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        let theory_name = compile_theory.as_str();
        if self.stack.len() > self.limits.max_depth {
            return Err(CompileGraphError::NestedLimit(format!(
                "{theory_name}.{definition}"
            )));
        }

        let current = DefinitionRef::new(compile_theory.clone(), definition);
        if let Some(index) = self.stack.iter().position(|entry| entry == &current) {
            // For now CompileGraph construction rejects cyclic nested definitions.
            // We may relax this later and render recursive definitions with
            // back-references instead of expanding them.
            let mut cycle = self.stack[index..]
                .iter()
                .map(DefinitionRef::label)
                .collect::<Vec<_>>();
            cycle.push(current.label());
            return Err(CompileGraphError::NestedCycle { cycle });
        }

        self.with_region_on_stack(current, |builder| {
            let theory = builder.compile_theory(&compile_theory)?;
            let definition_key = parse_operation(definition)?;
            let body = builder.compile_definition_body(&compile_theory, theory, &definition_key)?;
            let children = builder.compile_child_regions(&compile_theory, &body.graph)?;

            Ok(CompileGraph {
                theory: compile_theory,
                definition_name: definition.to_string(),
                graph: body.graph,
                source_variable_names: body.source_variable_names,
                children,
            })
        })
    }

    fn with_region_on_stack<T>(
        &mut self,
        current: DefinitionRef,
        build: impl FnOnce(&mut Self) -> Result<T, CompileGraphError>,
    ) -> Result<T, CompileGraphError> {
        self.stack.push(current);
        let result = build(self);
        self.stack.pop();
        result
    }

    fn compile_theory(&self, compile_theory: &CompileTheory) -> Result<&Theory, CompileGraphError> {
        self.theory_by_name(compile_theory.as_str())
    }

    fn theory_by_name(&self, theory_name: &str) -> Result<&Theory, CompileGraphError> {
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

    fn compile_definition_body(
        &self,
        compile_theory: &CompileTheory,
        theory: &Theory,
        definition_key: &Operation,
    ) -> Result<CompiledBody, CompileGraphError> {
        let loaded_definition = load_theory_definition(theory, definition_key)?;
        let graph = self.inline_selected_local_definitions(
            compile_theory,
            theory,
            loaded_definition.graph,
            definition_key,
        )?;
        let graph = typecheck_graph(
            theory,
            definition_key.as_str(),
            loaded_definition.source_type_map,
            loaded_definition.target_type_map,
            graph,
        )?;

        Ok(CompiledBody {
            graph: graph.to_strict(),
            source_variable_names: loaded_definition.source_variable_names,
        })
    }

    fn compile_child_regions(
        &mut self,
        compile_theory: &CompileTheory,
        graph: &StrictTypedGraph,
    ) -> Result<Vec<NestedCompileGraph>, CompileGraphError> {
        let mut seen = HashSet::new();
        let mut children = Vec::new();

        for operation in graph.h.x.0.iter() {
            let operation_name = operation.to_string();
            if !seen.insert(operation_name.clone()) {
                continue;
            }

            match self.classify_operation(compile_theory, &operation_name)? {
                OperationRole::Primitive => {}
                OperationRole::LocalDefinition => {
                    children.push(NestedCompileGraph {
                        operation: operation_name.clone(),
                        graph: self.compile_region(compile_theory.clone(), &operation_name)?,
                    });
                }
                OperationRole::CrossTheoryDefinition {
                    source_theory,
                    local_name,
                } => {
                    let graph = self.compile_cross_theory_child(source_theory, &local_name)?;
                    children.push(NestedCompileGraph {
                        operation: operation_name,
                        graph,
                    });
                }
            }
        }

        Ok(children)
    }

    fn inline_selected_local_definitions(
        &self,
        compile_theory: &CompileTheory,
        theory: &Theory,
        mut graph: DefinitionGraph,
        definition_key: &Operation,
    ) -> Result<DefinitionGraph, CompileGraphError> {
        let definitions = self.inline_policy.definitions_for(compile_theory, theory)?;
        if definitions.is_empty() {
            return Ok(graph);
        }

        for _ in 0..self.limits.max_inline_iterations {
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

    fn classify_operation(
        &self,
        compile_theory: &CompileTheory,
        operation_name: &str,
    ) -> Result<OperationRole, CompileGraphError> {
        if is_qualified_operation(operation_name) {
            return self.classify_cross_theory_operation(compile_theory, operation_name);
        }

        self.classify_local_operation(compile_theory, operation_name)
    }

    fn classify_local_operation(
        &self,
        compile_theory: &CompileTheory,
        operation_name: &str,
    ) -> Result<OperationRole, CompileGraphError> {
        let theory = self.compile_theory(compile_theory)?;
        if !definition_exists(theory, operation_name)? {
            return Ok(OperationRole::Primitive);
        }

        let qualified_name = format!("{compile_theory}.{operation_name}");
        if self.inline_policy.should_inline(&qualified_name) {
            Ok(OperationRole::Primitive)
        } else {
            Ok(OperationRole::LocalDefinition)
        }
    }

    fn classify_cross_theory_operation(
        &self,
        compile_theory: &CompileTheory,
        operation_name: &str,
    ) -> Result<OperationRole, CompileGraphError> {
        self.cross_theory_region_source(compile_theory, operation_name)
            .map(
                |(source_theory, local_name)| OperationRole::CrossTheoryDefinition {
                    source_theory,
                    local_name: local_name.to_string(),
                },
            )
            .or_else(|error| match error {
                // Dotted primitive names are allowed. For example, a backend
                // primitive may be named `gpu.view.linearize` without being a
                // data/control extension boundary.
                CompileGraphError::UnknownTheory(_) => Ok(OperationRole::Primitive),
                other => Err(other),
            })
    }

    fn cross_theory_region_source<'a>(
        &self,
        compile_theory: &CompileTheory,
        operation_name: &'a str,
    ) -> Result<(CompileTheory, &'a str), CompileGraphError> {
        let Some((extension_prefix, local_name)) = operation_name.split_once('.') else {
            return Err(CompileGraphError::UnknownOperation(
                operation_name.to_string(),
            ));
        };
        let Some(extension) = self
            .config
            .extension_for_target_and_prefix(compile_theory.as_str(), extension_prefix)
        else {
            return Err(CompileGraphError::UnknownTheory(
                extension_prefix.to_string(),
            ));
        };
        Ok((CompileTheory::parse(extension.source)?, local_name))
    }

    fn compile_cross_theory_child(
        &mut self,
        source_theory: CompileTheory,
        local_name: &str,
    ) -> Result<CompileGraph, CompileGraphError> {
        let native_cross_theory = self.compile_theory(&source_theory)?;
        if definition_exists(native_cross_theory, local_name)? {
            self.compile_region(source_theory, local_name)
        } else {
            compile_primitive_child_graph(native_cross_theory, source_theory, local_name)
        }
    }
}

#[derive(Clone, Copy)]
struct InlinePolicy {
    local_definitions: &'static [&'static str],
}

impl Default for InlinePolicy {
    fn default() -> Self {
        Self {
            local_definitions: INLINE_LOCAL_DEFINITIONS,
        }
    }
}

impl InlinePolicy {
    fn should_inline(&self, qualified_name: &str) -> bool {
        self.local_definitions.contains(&qualified_name)
    }

    fn definitions_for(
        &self,
        compile_theory: &CompileTheory,
        theory: &Theory,
    ) -> Result<HashMap<Operation, DefinitionGraph>, CompileGraphError> {
        let Theory::Theory { arrows, .. } = theory else {
            return Err(CompileGraphError::NotUserTheory(compile_theory.to_string()));
        };

        Ok(arrows
            .iter()
            .filter_map(|(name, arrow)| {
                let qualified_name = format!("{compile_theory}.{name}");
                self.should_inline(&qualified_name)
                    .then(|| arrow.definition.clone().map(|graph| (name.clone(), graph)))
                    .flatten()
            })
            .collect())
    }
}

fn load_theory_definition(
    theory: &Theory,
    definition_key: &Operation,
) -> Result<LoadedTheoryDefinition, CompileGraphError> {
    let arrow = theory
        .get_arrow(definition_key)
        .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
    let graph = arrow
        .definition
        .clone()
        .ok_or_else(|| CompileGraphError::UnknownDefinition(definition_key.to_string()))?;
    let source_variable_names = arrow
        .raw
        .definition
        .as_ref()
        .map(|definition| source_variable_names(theory, definition))
        .transpose()?
        .unwrap_or_default();

    Ok(LoadedTheoryDefinition {
        source_type_map: arrow.type_maps.0.clone(),
        target_type_map: arrow.type_maps.1.clone(),
        graph,
        source_variable_names,
    })
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

fn typecheck_graph(
    theory: &Theory,
    definition: &str,
    source: DefinitionGraph,
    target: DefinitionGraph,
    mut graph: DefinitionGraph,
) -> Result<TypedGraph, CompileGraphError> {
    let node_types = check(theory, source, target, &mut graph).map_err(|error| {
        CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error,
        }
    })?;
    let graph = graph
        .with_nodes(|_| node_types)
        .ok_or_else(|| CompileGraphError::Typecheck {
            definition: definition.to_string(),
            error: metacat::check::Error::InvalidTypeMaps,
        })?;

    Ok(graph)
}

fn definition_exists(theory: &Theory, local_name: &str) -> Result<bool, CompileGraphError> {
    let operation = parse_operation(local_name)?;
    Ok(theory
        .get_arrow(&operation)
        .and_then(|arrow| arrow.definition.as_ref())
        .is_some())
}

fn is_qualified_operation(operation_name: &str) -> bool {
    operation_name.split_once('.').is_some()
}

fn compile_primitive_child_graph(
    theory: &Theory,
    compile_theory: CompileTheory,
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
    let graph = typecheck_graph(
        theory,
        local_name,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        graph,
    )?;

    Ok(CompileGraph {
        theory: compile_theory,
        definition_name: local_name.to_string(),
        graph: graph.to_strict(),
        source_variable_names: HashMap::new(),
        children: Vec::new(),
    })
}

fn source_variable_names(
    theory: &Theory,
    definition: &Hexpr,
) -> Result<HashMap<usize, String>, CompileGraphError> {
    let signature = theory.local_signature();
    let graph = VariableNameInterpreter::new(&signature)
        .interpret(definition)
        .map_err(|_error| CompileGraphError::Typecheck {
            definition: "<variable names>".to_string(),
            error: metacat::check::Error::InvalidTypeMaps,
        })?;
    let quotient = graph.hypergraph.coequalizer();
    let mut names = HashMap::new();
    for (node, name) in graph.hypergraph.nodes.iter().enumerate() {
        let Some(name) = name else {
            continue;
        };
        names
            .entry(quotient.table[node])
            .or_insert_with(|| name.clone());
    }
    Ok(names)
}

struct VariableNameInterpreter<'a, S> {
    signature: &'a S,
    graph: LaxOpenHypergraph<Option<String>, Operation>,
    variables: HashMap<Variable, NodeId>,
}

impl<'a, S> VariableNameInterpreter<'a, S>
where
    S: Signature<Arr = Operation>,
{
    fn new(signature: &'a S) -> Self {
        Self {
            signature,
            graph: LaxOpenHypergraph::empty(),
            variables: HashMap::new(),
        }
    }

    fn interpret(
        mut self,
        hexpr: &Hexpr,
    ) -> Result<LaxOpenHypergraph<Option<String>, Operation>, hexpr::interpret::Error<S::Error>>
    {
        let (sources, targets) = self.interpret_stack(hexpr)?;
        self.graph.sources = sources;
        self.graph.targets = targets;
        Ok(self.graph)
    }

    fn interpret_stack(
        &mut self,
        hexpr: &Hexpr,
    ) -> Result<Interface, hexpr::interpret::Error<S::Error>> {
        match hexpr {
            Hexpr::Composition(hexprs) => self.interpret_composition(hexprs),
            Hexpr::Tensor(hexprs) => self.interpret_tensor(hexprs),
            Hexpr::Operation(op) => self.interpret_operation(op),
            Hexpr::Frobenius { sources, targets } => {
                let sources = self.frobenius_variables(sources);
                let targets = self.frobenius_variables(targets);
                Ok((sources, targets))
            }
        }
    }

    fn interpret_composition(
        &mut self,
        hexprs: &[Hexpr],
    ) -> Result<Interface, hexpr::interpret::Error<S::Error>> {
        let mut iter = hexprs.iter();
        let Some(mut current) = iter.next() else {
            return Ok((vec![], vec![]));
        };
        let (sources, mut current_targets) = self.interpret_stack(current)?;

        for next in iter {
            let (next_sources, next_targets) = self.interpret_stack(next)?;
            if current_targets.len() != next_sources.len() {
                return Err(hexpr::interpret::Error::Composition(
                    current.clone(),
                    next.clone(),
                ));
            }
            for (&target, &source) in current_targets.iter().zip(&next_sources) {
                self.graph.unify(target, source);
            }
            current_targets = next_targets;
            current = next;
        }

        Ok((sources, current_targets))
    }

    fn interpret_tensor(
        &mut self,
        hexprs: &[Hexpr],
    ) -> Result<Interface, hexpr::interpret::Error<S::Error>> {
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for hexpr in hexprs {
            let (next_sources, next_targets) = self.interpret_stack(hexpr)?;
            sources.extend(next_sources);
            targets.extend(next_targets);
        }
        Ok((sources, targets))
    }

    fn interpret_operation(
        &mut self,
        op: &hexpr::Operation,
    ) -> Result<Interface, hexpr::interpret::Error<S::Error>> {
        let arr = self
            .signature
            .try_parse_op(op)
            .map_err(|error| hexpr::interpret::Error::Signature(op.clone(), error))?;
        let (sources, targets) = self.signature.profile(&arr);
        let sources = vec![None; sources.len()];
        let targets = vec![None; targets.len()];
        let (_, interface) = self.graph.new_operation(arr, sources, targets);
        Ok(interface)
    }

    fn frobenius_variables(&mut self, variables: &[Variable]) -> Vec<NodeId> {
        variables
            .iter()
            .map(|variable| {
                if let Some(node) = self.variables.get(variable) {
                    *node
                } else {
                    let node = self.graph.new_node(Some(variable.to_string()));
                    self.variables.insert(variable.clone(), node);
                    node
                }
            })
            .collect()
    }
}

fn parse_operation(name: &str) -> Result<Operation, CompileGraphError> {
    name.parse()
        .map_err(|_| CompileGraphError::InvalidDefinition(name.to_string()))
}
