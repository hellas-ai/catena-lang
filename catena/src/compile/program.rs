use std::collections::HashMap;

use open_hypergraphs::lax::{NodeId, OpenHypergraph};
use thiserror::Error;

use crate::{
    compile::{CompileGraph, CompileTheory},
    lang::Obj,
    structured::{
        StructuredError, cfg,
        cfg::Cfg,
        ir::{Primitive, Stmt},
    },
};

#[derive(Debug, Error)]
pub enum ProgramCompileError {
    #[error("failed to build cfg: {0}")]
    Structure(#[from] StructuredError),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub entry: DefinitionId,
    pub definitions: HashMap<DefinitionId, Definition>,
}

impl Program {
    pub fn entry_definition(&self) -> &Definition {
        self.definitions
            .get(&self.entry)
            .expect("entry definition must exist")
    }
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: DefinitionId,
    pub name: String,
    pub params: Vec<VariableId>,
    pub returns: Vec<VariableId>,
    pub context: Context,
    pub body: Cfg,
}

#[derive(Debug, Clone)]
pub struct Context {
    variables: HashMap<VariableId, Variable>,
}

impl Context {
    pub fn new(variables: HashMap<VariableId, Variable>) -> Self {
        Self { variables }
    }

    pub fn variable(&self, id: VariableId) -> Option<&Variable> {
        self.variables.get(&id)
    }

    pub fn variables(&self) -> impl Iterator<Item = &Variable> {
        self.variables.values()
    }
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub id: VariableId,
    pub name: String,
    pub ty: Obj,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefinitionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(pub usize);

pub fn compile_program_from_graph(
    compile_graph: &CompileGraph,
) -> Result<Program, ProgramCompileError> {
    let mut definitions = HashMap::new();
    let mut next_id = 0;
    let entry = build_definition(compile_graph, &mut next_id, &mut definitions)?;
    Ok(Program { entry, definitions })
}

fn build_definition(
    compile_graph: &CompileGraph,
    next_id: &mut usize,
    definitions: &mut HashMap<DefinitionId, Definition>,
) -> Result<DefinitionId, ProgramCompileError> {
    let id = DefinitionId(*next_id);
    *next_id += 1;

    let cfg_context = cfg_context(compile_graph);
    let context = context_for_cfg_context(compile_graph, &cfg_context);
    let semantics = ProgramSemantics;
    let cfg_context = cfg_context.with_variables(variables_for_context(&context));
    let body = match cfg_context.kind() {
        cfg::GraphKind::Data => cfg::Cfg::from_dataflow_context(&cfg_context, &semantics)?,
        cfg::GraphKind::Control => cfg::Cfg::from_control_context(&cfg_context, &semantics)?,
    };

    definitions.insert(
        id,
        Definition {
            id,
            name: compile_graph.definition.clone(),
            params: cfg_context
                .graph()
                .sources
                .iter()
                .map(|node| VariableId(node.0))
                .collect(),
            returns: cfg_context
                .graph()
                .targets
                .iter()
                .map(|node| VariableId(node.0))
                .collect(),
            context,
            body,
        },
    );

    for child in &compile_graph.children {
        build_definition(&child.graph, next_id, definitions)?;
    }

    Ok(id)
}

fn context_for_cfg_context(compile_graph: &CompileGraph, context: &cfg::BuildContext) -> Context {
    let mut used_names = HashMap::new();
    Context::new(
        context
            .graph()
            .hypergraph
            .nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, ty)| {
                let id = VariableId(index);
                let name = variable_name(index, compile_graph, &mut used_names);
                (id, Variable { id, name, ty })
            })
            .collect(),
    )
}

fn variable_name(
    index: usize,
    compile_graph: &CompileGraph,
    used_names: &mut HashMap<String, usize>,
) -> String {
    let base = compile_graph
        .variable_names
        .get(&index)
        .map(|name| sanitize_ident(name))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| format!("w{index}"));
    unique_name(base, used_names)
}

fn unique_name(base: String, used_names: &mut HashMap<String, usize>) -> String {
    let count = used_names.entry(base.clone()).or_insert(0);
    if *count == 0 {
        *count += 1;
        return base;
    }
    let name = format!("{base}{count}");
    *count += 1;
    name
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

fn variables_for_context(context: &Context) -> HashMap<NodeId, String> {
    context
        .variables()
        .map(|variable| (NodeId(variable.id.0), variable.name.clone()))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct ProgramSemantics;

impl cfg::ArrowSemantics for ProgramSemantics {
    fn statements(&self, arrow: &cfg::ArrowInstance) -> Vec<Stmt> {
        if arrow.op == "gpu.sync" {
            return vec![Stmt::Barrier];
        }
        let outputs = if arrow.branch_arity > 1 {
            vec![branch_tag(arrow), branch_payload(arrow)]
        } else if arrow.op.starts_with("data.") {
            arrow.outputs.clone()
        } else {
            Vec::new()
        };
        vec![Stmt::Primitive(Primitive {
            name: arrow.op.clone(),
            inputs: arrow.inputs.clone(),
            outputs,
            code: String::new(),
        })]
    }

    fn branch_condition_rhs(&self, arrow: &cfg::ArrowInstance, output: usize) -> String {
        format!("{} == {output}", branch_tag(arrow))
    }
}

fn branch_tag(arrow: &cfg::ArrowInstance) -> String {
    format!("b{}", arrow.id)
}

fn branch_payload(arrow: &cfg::ArrowInstance) -> String {
    format!("p{}", arrow.id)
}

fn cfg_context(graph: &CompileGraph) -> cfg::BuildContext {
    cfg::BuildContext::new(
        graph_kind(&graph.theory),
        OpenHypergraph::from_strict(graph.typed_graph.clone()),
        graph
            .children
            .iter()
            .map(|child| (child.operation.clone(), cfg_context(&child.graph)))
            .collect(),
    )
}

fn graph_kind(theory: &CompileTheory) -> cfg::GraphKind {
    match theory {
        CompileTheory::Data => cfg::GraphKind::Data,
        CompileTheory::Control => cfg::GraphKind::Control,
    }
}
