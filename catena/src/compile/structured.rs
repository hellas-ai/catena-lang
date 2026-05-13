use hexpr::Operation;
use metacat::{
    check::check,
    theory::{TheoryId, TheorySet},
};
use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};
use thiserror::Error;

use crate::{
    compile::CompileGraph,
    lang::{Arr, Obj},
    pass::{erase::Erase, forget_loopback::ForgetLoopback},
    structured::{
        StructuredError, cfg,
        ir::{EntryPoint, Program, Stmt},
        ramsey,
    },
};

#[derive(Debug, Error)]
pub enum StructuredCompileError {
    #[error("unknown theory `{0}`")]
    UnknownTheory(String),
    #[error("invalid entry arrow `{0}`")]
    InvalidEntry(String),
    #[error("unknown entry arrow `{0}`")]
    UnknownEntry(String),
    #[error("entry arrow `{0}` has no definition")]
    MissingDefinition(String),
    #[error("entry arrow `{entry}` failed typecheck: {detail:?}")]
    EntryTypecheck {
        entry: String,
        detail: metacat::check::Error<Operation>,
    },
    #[error("failed to normalize entry graph after typecheck: {detail}")]
    Normalize { detail: String },
    #[error("failed to structure control graph: {0}")]
    Structure(#[from] StructuredError),
}

pub fn compile_structured_program_from_graph(
    theory_set: &TheorySet,
    theory: &str,
    entry: &str,
    compile_graph: &CompileGraph,
) -> Result<Program, StructuredCompileError> {
    let entry_graph = typed_definition_graph(theory_set, theory, entry)?;
    let entry_graph = normalize_structured_graph(&entry_graph)?;
    let control = GenericControl;
    let context = cfg::Context::new(compile_graph);
    let cfg = cfg::Cfg::from_hypergraph(&entry_graph, &context, &control)?;
    let body = ramsey::structure(cfg)?;
    Ok(program(entry, body))
}

fn program(entry: &str, body: Vec<Stmt>) -> Program {
    Program {
        name: sanitize_ident(entry),
        entry: EntryPoint {
            name: sanitize_ident(entry),
            params: Vec::new(),
        },
        body,
    }
}

#[derive(Debug, Clone, Copy)]
struct GenericControl;

impl cfg::ArrowSemantics for GenericControl {
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
        vec![Stmt::Primitive(crate::structured::ir::Primitive {
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

fn normalize_structured_graph(
    graph: &OpenHypergraph<Obj, Arr>,
) -> Result<OpenHypergraph<Obj, Arr>, StructuredCompileError> {
    let loopback = ForgetLoopback::default_control();
    let mut graph = Erase::with_value(loopback.config().value).map_arrow(graph);
    quotient_normalized(&mut graph)?;
    graph = loopback.map_arrow(&graph);
    quotient_normalized(&mut graph)?;
    Ok(graph)
}

fn quotient_normalized(graph: &mut OpenHypergraph<Obj, Arr>) -> Result<(), StructuredCompileError> {
    graph
        .quotient()
        .map_err(|detail| StructuredCompileError::Normalize {
            detail: format!("{detail:?}"),
        })?;
    Ok(())
}

fn typed_definition_graph(
    theory_set: &TheorySet,
    theory_name: &str,
    entry: &str,
) -> Result<OpenHypergraph<Obj, Arr>, StructuredCompileError> {
    let theory_id = TheoryId(
        theory_name
            .parse()
            .map_err(|_| StructuredCompileError::UnknownTheory(theory_name.to_string()))?,
    );
    let theory = theory_set
        .theories
        .get(&theory_id)
        .ok_or_else(|| StructuredCompileError::UnknownTheory(theory_name.to_string()))?;

    let entry_key: Operation = entry
        .parse()
        .map_err(|_| StructuredCompileError::InvalidEntry(entry.to_string()))?;
    let arrow = theory
        .get_arrow(&entry_key)
        .ok_or_else(|| StructuredCompileError::UnknownEntry(entry.to_string()))?;
    let mut graph = arrow
        .definition
        .clone()
        .ok_or_else(|| StructuredCompileError::MissingDefinition(entry.to_string()))?;

    let node_types = check(
        theory,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        &mut graph,
    )
    .map_err(|detail| StructuredCompileError::EntryTypecheck {
        entry: entry.to_string(),
        detail,
    })?;

    graph
        .with_nodes(|_| node_types)
        .ok_or_else(|| StructuredCompileError::EntryTypecheck {
            entry: entry.to_string(),
            detail: metacat::check::Error::InvalidTypeMaps,
        })
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
