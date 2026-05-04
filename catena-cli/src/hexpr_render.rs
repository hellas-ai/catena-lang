use std::collections::HashSet;

use catena::compile::ArrowType;
use hexpr::Operation;
use open_hypergraphs::category::Arrow;
use open_hypergraphs::lax::{NodeId, OpenHypergraph};

// This is HExpr pretty-printing for interpreted object maps. It may belong in
// hexpr/metacat eventually, once the desired notation is stable.
pub fn render_arrow_declaration(arrow_type: &ArrowType) -> String {
    format!(
        "(arr {} : {} -> {})",
        arrow_type.name,
        render_object_map(&arrow_type.source),
        render_object_map(&arrow_type.target)
    )
}

pub fn render_object_map(map: &OpenHypergraph<(), Operation>) -> String {
    let mut map = map.clone();
    let _ = map.quotient();
    let map = &map;
    let vars = source_vars(map);
    match map.target().len() {
        0 => "[]".to_string(),
        1 => render_target(map, map.targets[0], &vars),
        _ => {
            let targets = map
                .targets
                .iter()
                .map(|node| render_node(map, *node, &vars, &mut HashSet::new()))
                .collect::<Vec<_>>();
            render_spider(&vars, &targets)
        }
    }
}

fn render_target(map: &OpenHypergraph<(), Operation>, node: NodeId, vars: &[String]) -> String {
    render_node(map, node, vars, &mut HashSet::new())
}

fn render_edge(
    map: &OpenHypergraph<(), Operation>,
    edge_index: usize,
    vars: &[String],
    seen: &mut HashSet<NodeId>,
) -> String {
    let op = &map.hypergraph.edges[edge_index];
    let adjacency = &map.hypergraph.adjacency[edge_index];
    if adjacency.sources.is_empty() {
        return op.to_string();
    }

    let inputs = adjacency
        .sources
        .iter()
        .map(|node| render_node(map, *node, vars, seen))
        .collect::<Vec<_>>();
    format!("({} {op})", render_spider(vars, &inputs))
}

fn render_node(
    map: &OpenHypergraph<(), Operation>,
    node: NodeId,
    vars: &[String],
    seen: &mut HashSet<NodeId>,
) -> String {
    if let Some(var) = map
        .sources
        .iter()
        .position(|source| *source == node)
        .map(|index| vars[index].clone())
    {
        return var;
    }

    if !seen.insert(node) {
        return format!("n{}", node.0);
    }

    let rendered = producer_edge(map, node)
        .map(|edge_index| render_edge(map, edge_index, vars, seen))
        .or_else(|| {
            object_edge_at_node(map, node)
                .map(|edge_index| map.hypergraph.edges[edge_index].to_string())
        })
        .unwrap_or_else(|| format!("n{}", node.0));
    seen.remove(&node);
    rendered
}

fn render_spider(sources: &[String], targets: &[String]) -> String {
    if sources == targets {
        format!("[{}]", sources.join(" "))
    } else if targets.is_empty() {
        format!("[{} .]", sources.join(" "))
    } else {
        format!("[{} . {}]", sources.join(" "), targets.join(" "))
    }
}

fn source_vars(map: &OpenHypergraph<(), Operation>) -> Vec<String> {
    (0..map.source().len())
        .map(|index| format!("x{index}"))
        .collect()
}

fn producer_edge(map: &OpenHypergraph<(), Operation>, node: NodeId) -> Option<usize> {
    map.hypergraph
        .adjacency
        .iter()
        .position(|edge| edge.targets.contains(&node))
}

fn object_edge_at_node(map: &OpenHypergraph<(), Operation>, node: NodeId) -> Option<usize> {
    map.hypergraph
        .adjacency
        .iter()
        .position(|edge| edge.sources.is_empty() && edge.targets.contains(&node))
}
