use std::collections::HashMap;

use crate::{compile::CompileGraph, lang::Obj};
use graphviz_rust::{
    cmd::{CommandArg, Format},
    dot_structures::{
        Attribute, Edge, EdgeTy, Graph, GraphAttributes, Id, Node, NodeId as DotNodeId, Port, Stmt,
        Subgraph, Vertex,
    },
    exec,
    printer::PrinterContext,
};
use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::strict::vec::OpenHypergraph;

// This is intentionally custom instead of delegating to open-hypergraphs-dot:
// Catena graphs need hierarchical DOT clusters, cluster boundary edges, hidden
// interface anchors, and theory-specific nesting rules that a flat renderer
// cannot express without substantial post-processing.
pub fn nested_svg(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    let mut renderer = NestedDotRenderer::default();
    exec(
        renderer.render(graph),
        &mut PrinterContext::default(),
        vec![CommandArg::Format(Format::Svg)],
    )
}

#[derive(Default)]
struct NestedDotRenderer {
    next_graph_id: usize,
}

struct RenderedInterface {
    cluster_id: String,
    sources: Vec<String>,
    targets: Vec<String>,
}

struct ParentInterface {
    source_labels: Vec<String>,
    target_labels: Vec<String>,
}

impl NestedDotRenderer {
    fn render(&mut self, graph: &CompileGraph) -> Graph {
        let (cluster, _) = self.render_cluster(graph, &qualified_name(graph), None);
        Graph::DiGraph {
            id: id("G"),
            strict: false,
            stmts: vec![
                Stmt::GAttribute(GraphAttributes::Graph(vec![
                    attr_plain("rankdir", "TB"),
                    attr_quoted("bgcolor", "#4a4a4a"),
                    attr_plain("compound", "true"),
                ])),
                Stmt::GAttribute(GraphAttributes::Node(vec![
                    attr_quoted("fontcolor", "white"),
                    attr_quoted("color", "white"),
                ])),
                Stmt::GAttribute(GraphAttributes::Edge(vec![
                    attr_quoted("fontcolor", "white"),
                    attr_quoted("color", "white"),
                ])),
                Stmt::Subgraph(cluster),
            ],
        }
    }

    fn render_cluster(
        &mut self,
        graph: &CompileGraph,
        label: &str,
        parent_interface: Option<ParentInterface>,
    ) -> (Subgraph, RenderedInterface) {
        let graph_id = self.next_id();
        let prefix = format!("g{graph_id}");
        let cluster_id = cluster_id(graph_id);
        let mut stmts = vec![
            Stmt::Attribute(attr_quoted("label", label)),
            Stmt::Attribute(attr_quoted("color", "white")),
            Stmt::Attribute(attr_quoted("fontcolor", "white")),
            Stmt::Attribute(attr_plain("style", "rounded")),
        ];
        let children = graph
            .children
            .iter()
            .map(|child| (child.operation.as_str(), &child.graph))
            .collect::<HashMap<_, _>>();

        if let Some(interface) = parent_interface.as_ref() {
            self.render_external_interface(&prefix, interface, &mut stmts);
        }
        self.render_nodes(&prefix, &graph.graph, &mut stmts);
        self.render_boundary(&prefix, &graph.graph, &mut stmts);

        for edge_index in 0..graph.graph.h.x.0.len() {
            let operation = &graph.graph.h.x.0[edge_index];
            if let Some(child) = children.get(operation.to_string().as_str()) {
                let sources = edge_sources(&graph.graph, edge_index);
                let targets = edge_targets(&graph.graph, edge_index);
                let (child_cluster, child_interface) = self.render_cluster(
                    child,
                    &operation.to_string(),
                    Some(ParentInterface {
                        source_labels: sources
                            .iter()
                            .map(|node| object_label(&graph.graph.h.w.0[*node]))
                            .collect(),
                        target_labels: targets
                            .iter()
                            .map(|node| object_label(&graph.graph.h.w.0[*node]))
                            .collect(),
                    }),
                );
                stmts.push(Stmt::Subgraph(child_cluster));
                self.render_nested_connections(
                    &prefix,
                    &graph.graph,
                    edge_index,
                    &child_interface,
                    &mut stmts,
                );
            } else {
                self.render_edge_box(&prefix, &graph.graph, edge_index, operation, &mut stmts);
            }
        }

        let rendered_interface = if let Some(interface) = parent_interface {
            RenderedInterface {
                cluster_id: cluster_id.clone(),
                sources: (0..interface.source_labels.len())
                    .map(|index| interface_source_id(&prefix, index))
                    .collect(),
                targets: (0..interface.target_labels.len())
                    .map(|index| interface_target_id(&prefix, index))
                    .collect(),
            }
        } else {
            RenderedInterface {
                cluster_id: cluster_id.clone(),
                sources: graph
                    .graph
                    .s
                    .table
                    .iter()
                    .map(|node| node_id(&prefix, *node))
                    .collect(),
                targets: graph
                    .graph
                    .t
                    .table
                    .iter()
                    .map(|node| node_id(&prefix, *node))
                    .collect(),
            }
        };

        (
            Subgraph {
                id: id(&cluster_id),
                stmts,
            },
            rendered_interface,
        )
    }

    fn render_external_interface(
        &self,
        prefix: &str,
        interface: &ParentInterface,
        stmts: &mut Vec<Stmt>,
    ) {
        for index in 0..interface.source_labels.len() {
            self.render_invisible_interface_node(&interface_source_id(prefix, index), stmts);
        }
        for index in 0..interface.target_labels.len() {
            self.render_invisible_interface_node(&interface_target_id(prefix, index), stmts);
        }
    }

    fn render_invisible_interface_node(&self, node_id: &str, stmts: &mut Vec<Stmt>) {
        stmts.push(node_stmt(
            node_id,
            vec![
                attr_plain("shape", "point"),
                attr_plain("style", "invis"),
                attr_quoted("label", ""),
                attr_plain("width", "0.01"),
                attr_plain("height", "0.01"),
            ],
        ));
    }

    fn render_nodes(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<Obj, Operation>,
        stmts: &mut Vec<Stmt>,
    ) {
        for node_index in 0..graph.h.w.0.len() {
            let label = object_label(&graph.h.w.0[node_index]);
            stmts.push(node_stmt(
                &node_id(prefix, node_index),
                vec![attr_plain("shape", "point"), attr_quoted("xlabel", &label)],
            ));
        }
    }

    fn render_boundary(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<Obj, Operation>,
        stmts: &mut Vec<Stmt>,
    ) {
        for (index, source) in graph.s.table.iter().enumerate() {
            stmts.push(boundary_node(&input_id(prefix, index)));
            stmts.push(edge_stmt(
                node_vertex(&input_id(prefix, index)),
                node_vertex(&node_id(prefix, *source)),
                vec![attr_plain("style", "dashed"), attr_plain("dir", "none")],
            ));
        }

        for (index, target) in graph.t.table.iter().enumerate() {
            stmts.push(boundary_node(&output_id(prefix, index)));
            stmts.push(edge_stmt(
                node_vertex(&node_id(prefix, *target)),
                node_vertex(&output_id(prefix, index)),
                vec![attr_plain("style", "dashed"), attr_plain("dir", "none")],
            ));
        }

        if !graph.s.table.is_empty() {
            stmts.push(rank_subgraph(
                "source",
                (0..graph.s.table.len())
                    .map(|index| input_id(prefix, index))
                    .collect(),
            ));
        }

        if !graph.t.table.is_empty() {
            stmts.push(rank_subgraph(
                "sink",
                (0..graph.t.table.len())
                    .map(|index| output_id(prefix, index))
                    .collect(),
            ));
        }
    }

    fn render_edge_box(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<Obj, Operation>,
        edge_index: usize,
        operation: &Operation,
        stmts: &mut Vec<Stmt>,
    ) {
        let edge_id = edge_id(prefix, edge_index);
        let sources = edge_sources(graph, edge_index);
        let targets = edge_targets(graph, edge_index);
        let label = record_label(operation, sources.len(), targets.len());

        stmts.push(node_stmt(
            &edge_id,
            vec![attr_quoted("label", &label), attr_plain("shape", "record")],
        ));

        for (source_index, source) in sources.iter().enumerate() {
            stmts.push(edge_stmt(
                node_vertex(&node_id(prefix, *source)),
                port_vertex(&edge_id, &format!("s_{source_index}")),
                vec![],
            ));
        }
        for (target_index, target) in targets.iter().enumerate() {
            stmts.push(edge_stmt(
                port_vertex(&edge_id, &format!("t_{target_index}")),
                node_vertex(&node_id(prefix, *target)),
                vec![],
            ));
        }
    }

    fn render_nested_connections(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<Obj, Operation>,
        edge_index: usize,
        child: &RenderedInterface,
        stmts: &mut Vec<Stmt>,
    ) {
        let sources = edge_sources(graph, edge_index);
        let targets = edge_targets(graph, edge_index);
        for (source, child_source) in sources.iter().zip(&child.sources) {
            stmts.push(edge_stmt(
                node_vertex(&node_id(prefix, *source)),
                node_vertex(child_source),
                vec![attr_plain("lhead", &child.cluster_id)],
            ));
        }
        for (child_target, target) in child.targets.iter().zip(&targets) {
            stmts.push(edge_stmt(
                node_vertex(child_target),
                node_vertex(&node_id(prefix, *target)),
                vec![attr_plain("ltail", &child.cluster_id)],
            ));
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.next_graph_id;
        self.next_graph_id += 1;
        id
    }
}

fn qualified_name(graph: &CompileGraph) -> String {
    format!("{}.{}", graph.theory, graph.definition_name)
}

fn node_id(prefix: &str, node: usize) -> String {
    format!("{prefix}_n_{node}")
}

fn edge_id(prefix: &str, edge_index: usize) -> String {
    format!("{prefix}_e_{edge_index}")
}

fn input_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_input_{index}")
}

fn output_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_output_{index}")
}

fn cluster_id(graph_id: usize) -> String {
    format!("cluster_{graph_id}")
}

fn edge_sources(graph: &OpenHypergraph<Obj, Operation>, edge_index: usize) -> Vec<usize> {
    graph
        .h
        .s
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|sources| sources.table.0)
        .unwrap_or_default()
}

fn edge_targets(graph: &OpenHypergraph<Obj, Operation>, edge_index: usize) -> Vec<usize> {
    graph
        .h
        .t
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|targets| targets.table.0)
        .unwrap_or_default()
}

pub(crate) fn object_label(object: &Obj) -> String {
    match object {
        Tree::Empty => "empty".to_string(),
        Tree::Leaf(index, _) => format!("x{index}"),
        Tree::Node(op, target_index, children) => {
            let inner = object_node_label(op, children);
            if *target_index == 0 {
                inner
            } else {
                format!("π{target_index}({inner})")
            }
        }
    }
}

fn object_node_label(op: &Operation, children: &[Obj]) -> String {
    match children.len() {
        0 => format!("{op}"),
        1 => format!("{op}({})", object_label(&children[0])),
        2 => {
            let op = format!("{op}");
            if op.starts_with(|c: char| c.is_alphanumeric()) {
                format!(
                    "{}({}, {})",
                    op,
                    object_label(&children[0]),
                    object_label(&children[1])
                )
            } else {
                format!(
                    "{} {op} {}",
                    object_label(&children[0]),
                    object_label(&children[1])
                )
            }
        }
        _ => {
            let args = children
                .iter()
                .map(object_label)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{op}({args})")
        }
    }
}

fn interface_source_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_source_{index}")
}

fn interface_target_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_target_{index}")
}

fn record_label(operation: &Operation, source_arity: usize, target_arity: usize) -> String {
    let sources = record_ports("s", source_arity);
    let targets = record_ports("t", target_arity);

    match (sources.is_empty(), targets.is_empty()) {
        (true, true) => operation.to_string(),
        (true, false) => format!("{} | {{ {targets} }}", operation),
        (false, true) => format!("{{ {sources} }} | {}", operation),
        (false, false) => format!("{{ {sources} }} | {} | {{ {targets} }}", operation),
    }
}

fn record_ports(prefix: &str, arity: usize) -> String {
    (0..arity)
        .map(|index| format!("<{prefix}_{index}>"))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn boundary_node(node_id: &str) -> Stmt {
    node_stmt(
        node_id,
        vec![
            attr_plain("shape", "point"),
            attr_quoted("label", ""),
            attr_plain("width", "0.05"),
            attr_plain("height", "0.05"),
        ],
    )
}

fn rank_subgraph(rank: &str, nodes: Vec<String>) -> Stmt {
    Stmt::Subgraph(Subgraph {
        id: Id::Anonymous(String::new()),
        stmts: std::iter::once(Stmt::Attribute(attr_plain("rank", rank)))
            .chain(nodes.into_iter().map(|node| node_stmt(&node, vec![])))
            .collect(),
    })
}

fn node_stmt(node_id: &str, attributes: Vec<Attribute>) -> Stmt {
    Stmt::Node(Node {
        id: dot_node_id(node_id),
        attributes,
    })
}

fn edge_stmt(source: Vertex, target: Vertex, attributes: Vec<Attribute>) -> Stmt {
    Stmt::Edge(Edge {
        ty: EdgeTy::Pair(source, target),
        attributes,
    })
}

fn node_vertex(node_id: &str) -> Vertex {
    Vertex::N(dot_node_id(node_id))
}

fn port_vertex(node_id: &str, port: &str) -> Vertex {
    Vertex::N(DotNodeId(
        id(node_id),
        Some(Port(None, Some(port.to_string()))),
    ))
}

fn dot_node_id(node_id: &str) -> DotNodeId {
    DotNodeId(id(node_id), None)
}

fn attr_plain(name: &str, value: &str) -> Attribute {
    Attribute(id(name), id(value))
}

fn attr_quoted(name: &str, value: &str) -> Attribute {
    Attribute(id(name), Id::Escaped(quoted(value)))
}

fn id(value: &str) -> Id {
    Id::Plain(value.to_string())
}

fn quoted(value: &str) -> String {
    format!("\"{}\"", escape_quoted(value))
}

fn escape_quoted(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            _ => vec![character],
        })
        .collect()
}
