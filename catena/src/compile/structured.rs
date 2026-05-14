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
    #[error("failed to normalize entry graph after typecheck: {detail}")]
    Normalize { detail: String },
    #[error("failed to structure control graph: {0}")]
    Structure(#[from] StructuredError),
}

pub fn compile_structured_program_from_graph(
    compile_graph: &CompileGraph,
) -> Result<Program, StructuredCompileError> {
    let graph = structured_graph(compile_graph)?;
    let control = GenericControl;
    let context = cfg::Context::new(&graph);
    let cfg = cfg::Cfg::from_context(&context, &control)?;
    let body = ramsey::structure(cfg)?;
    Ok(program(&compile_graph.definition, body))
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

fn structured_graph(graph: &CompileGraph) -> Result<cfg::Graph, StructuredCompileError> {
    let typed_graph = OpenHypergraph::from_strict(graph.typed_graph.clone());
    let graph = cfg::Graph {
        name: graph.definition.clone(),
        graph: normalize_structured_graph(&typed_graph)?,
        children: graph
            .children
            .iter()
            .map(|child| {
                Ok(cfg::ChildGraph {
                    operation: child.operation.clone(),
                    graph: structured_graph(&child.graph)?,
                })
            })
            .collect::<Result<Vec<_>, StructuredCompileError>>()?,
    };
    Ok(graph)
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
