use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};
use thiserror::Error;

use crate::{
    compile::{CompileGraph, graph::NestedCompileGraph},
    lang::{Arr, Obj},
    pass::{erase::Erase, forget_loopback::ForgetLoopback},
};

#[derive(Debug, Error)]
pub enum NormalizeGraphError {
    #[error("failed to normalize entry graph after typecheck: {detail}")]
    Normalize { detail: String },
}

pub fn normalize_graph(graph: &CompileGraph) -> Result<CompileGraph, NormalizeGraphError> {
    let graph_body = OpenHypergraph::from_strict(graph.graph.clone());
    Ok(CompileGraph {
        theory: graph.theory.clone(),
        definition_name: graph.definition_name.clone(),
        graph: normalize_graph_hypergraph(&graph_body)?.to_strict(),
        source_variable_names: graph.source_variable_names.clone(),
        children: graph
            .children
            .iter()
            .map(|child| {
                Ok(NestedCompileGraph {
                    operation: child.operation.clone(),
                    graph: normalize_graph(&child.graph)?,
                })
            })
            .collect::<Result<Vec<_>, NormalizeGraphError>>()?,
    })
}

fn normalize_graph_hypergraph(
    graph: &OpenHypergraph<Obj, Arr>,
) -> Result<OpenHypergraph<Obj, Arr>, NormalizeGraphError> {
    let loopback = ForgetLoopback::default_control();
    let mut graph = Erase::with_value(loopback.config().value).map_arrow(graph);
    quotient_normalized(&mut graph)?;
    graph = loopback.map_arrow(&graph);
    quotient_normalized(&mut graph)?;
    Ok(graph)
}

fn quotient_normalized(graph: &mut OpenHypergraph<Obj, Arr>) -> Result<(), NormalizeGraphError> {
    graph
        .quotient()
        .map_err(|detail| NormalizeGraphError::Normalize {
            detail: format!("{detail:?}"),
        })?;
    Ok(())
}
