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
    let typed_graph = OpenHypergraph::from_strict(graph.typed_graph.clone());
    Ok(CompileGraph {
        theory: graph.theory.clone(),
        definition: graph.definition.clone(),
        graph: graph.graph.clone(),
        typed_graph: normalize_graph_hypergraph(&typed_graph)?.to_strict(),
        variable_names: graph.variable_names.clone(),
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
