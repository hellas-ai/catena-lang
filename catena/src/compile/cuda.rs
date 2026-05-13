use hexpr::Operation;
use metacat::{
    check::check,
    theory::{RawTheorySet, TheoryId, TheorySet, ast::ParseRawError},
};
use open_hypergraphs::lax::OpenHypergraph;
use thiserror::Error;

mod domain;
mod render;

use domain::CudaTarget;

use crate::{
    check::check as check_elaborated,
    compile::{CompileConfig, CompileGraphError, GraphCompileOptions, compile_graph_with_options},
    elaborate::elaborate,
    lang::{Arr, Obj},
    pass::{erase::Erase, forget_loopback::ForgetLoopback},
    structured::{StructuredError, cfg, ramsey},
};
use open_hypergraphs::lax::functor::Functor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaEmit {
    Cuda,
    StructuredIr,
}

#[derive(Debug, Error)]
pub enum CudaCompileError {
    #[error("failed to parse source: {0}")]
    Parse(#[from] ParseRawError),
    #[error("failed to elaborate or typecheck source: {0}")]
    Check(#[from] crate::check::CheckError),
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
    #[error("failed to build compile graph: {0}")]
    CompileGraph(#[from] CompileGraphError),
    #[error("failed to normalize entry graph after typecheck: {detail}")]
    Normalize { detail: String },
    #[error("failed to structure control graph: {0}")]
    Structure(#[from] StructuredError),
}

pub fn compile_cuda_source(
    source: &str,
    theory: &str,
    entry: &str,
    emit: CudaEmit,
) -> Result<String, CudaCompileError> {
    let raw = RawTheorySet::from_text(source)?;
    let elaborated = elaborate(raw)?;
    let theory_set = check_elaborated(&elaborated)?;
    compile_cuda_checked(&theory_set, theory, entry, emit)
}

fn compile_cuda_checked(
    theory_set: &TheorySet,
    theory: &str,
    entry: &str,
    emit: CudaEmit,
) -> Result<String, CudaCompileError> {
    let compile_graph = compile_graph_with_options(
        theory_set,
        &CompileConfig::data_control(),
        theory,
        entry,
        GraphCompileOptions::default(),
    )?;
    let entry_graph = typed_definition_graph(theory_set, theory, entry)?;
    let entry_graph = normalize_structured_cuda_graph(&entry_graph)?;
    let target = CudaTarget::new(theory_set);
    let context = cfg::Context::new(&compile_graph);
    let cfg = cfg::Cfg::from_hypergraph(&entry_graph, &context, &target.control)?;
    let body = ramsey::structure(cfg)?;
    let program = target.program(entry, body);

    match emit {
        CudaEmit::Cuda => Ok(target.render_cuda_with_launch(&program)),
        CudaEmit::StructuredIr => Ok(program.render_ir()),
    }
}

fn normalize_structured_cuda_graph(
    graph: &OpenHypergraph<Obj, Arr>,
) -> Result<OpenHypergraph<Obj, Arr>, CudaCompileError> {
    let loopback = ForgetLoopback::default_control();
    let mut graph = Erase::with_value(loopback.config().value).map_arrow(graph);
    quotient_normalized(&mut graph)?;
    graph = loopback.map_arrow(&graph);
    quotient_normalized(&mut graph)?;
    Ok(graph)
}

fn quotient_normalized(graph: &mut OpenHypergraph<Obj, Arr>) -> Result<(), CudaCompileError> {
    graph
        .quotient()
        .map_err(|detail| CudaCompileError::Normalize {
            detail: format!("{detail:?}"),
        })?;
    Ok(())
}

fn typed_definition_graph(
    theory_set: &TheorySet,
    theory_name: &str,
    entry: &str,
) -> Result<OpenHypergraph<Obj, Arr>, CudaCompileError> {
    let theory_id = TheoryId(
        theory_name
            .parse()
            .map_err(|_| CudaCompileError::UnknownTheory(theory_name.to_string()))?,
    );
    let theory = theory_set
        .theories
        .get(&theory_id)
        .ok_or_else(|| CudaCompileError::UnknownTheory(theory_name.to_string()))?;

    let entry_key: Operation = entry
        .parse()
        .map_err(|_| CudaCompileError::InvalidEntry(entry.to_string()))?;
    let arrow = theory
        .get_arrow(&entry_key)
        .ok_or_else(|| CudaCompileError::UnknownEntry(entry.to_string()))?;
    let mut graph = arrow
        .definition
        .clone()
        .ok_or_else(|| CudaCompileError::MissingDefinition(entry.to_string()))?;

    let node_types = check(
        theory,
        arrow.type_maps.0.clone(),
        arrow.type_maps.1.clone(),
        &mut graph,
    )
    .map_err(|detail| CudaCompileError::EntryTypecheck {
        entry: entry.to_string(),
        detail,
    })?;

    graph
        .with_nodes(|_| node_types)
        .ok_or_else(|| CudaCompileError::EntryTypecheck {
            entry: entry.to_string(),
            detail: metacat::check::Error::InvalidTypeMaps,
        })
}
