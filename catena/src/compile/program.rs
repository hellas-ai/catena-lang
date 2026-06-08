use std::collections::HashMap;

use thiserror::Error;

use crate::{
    compile::{
        CompileGraph, CompileTheory,
        cfg::{self, Cfg, CfgError, CfgOptions},
    },
    lang::Obj,
};

#[derive(Debug, Error)]
pub enum ProgramCompileError {
    #[error("failed to build cfg: {0}")]
    Structure(#[from] CfgError),
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramCompileOptions {
    pub cfg: CfgOptions,
}

pub fn compile_program_from_graph(
    compile_graph: &CompileGraph,
) -> Result<Program, ProgramCompileError> {
    compile_program_from_graph_with_options(compile_graph, ProgramCompileOptions::default())
}

pub fn compile_program_from_graph_with_options(
    compile_graph: &CompileGraph,
    options: ProgramCompileOptions,
) -> Result<Program, ProgramCompileError> {
    let mut definitions = HashMap::new();
    let mut next_id = 0;
    let entry = build_definition(compile_graph, options, &mut next_id, &mut definitions)?;
    Ok(Program { entry, definitions })
}

fn build_definition(
    compile_graph: &CompileGraph,
    options: ProgramCompileOptions,
    next_id: &mut usize,
    definitions: &mut HashMap<DefinitionId, Definition>,
) -> Result<DefinitionId, ProgramCompileError> {
    let id = DefinitionId(*next_id);
    *next_id += 1;

    let context = context_for_graph(compile_graph);
    let body = cfg::build_cfg(compile_graph, options.cfg)?.cfg().clone();

    definitions.insert(
        id,
        Definition {
            id,
            name: compile_graph.definition_name.clone(),
            params: compile_graph
                .graph
                .s
                .table
                .iter()
                .map(|node| VariableId(*node))
                .collect(),
            returns: compile_graph
                .graph
                .t
                .table
                .iter()
                .map(|node| VariableId(*node))
                .collect(),
            context,
            body,
        },
    );

    for child in &compile_graph.children {
        if matches!(child.graph.theory, CompileTheory::Data) {
            build_definition(&child.graph, options, next_id, definitions)?;
        }
    }

    Ok(id)
}

fn context_for_graph(compile_graph: &CompileGraph) -> Context {
    let mut used_names = HashMap::new();
    Context::new(
        compile_graph
            .graph
            .h
            .w
            .0
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
        .source_variable_names
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
