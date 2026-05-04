use std::collections::HashMap;

use catena::compile::CompileGraph;
use graphviz_rust::{
    cmd::{CommandArg, Format},
    exec_dot,
};
use metacat::theory::OperationKey;
use open_hypergraphs::lax::{NodeId, OpenHypergraph};

// open-hypergraphs-dot handles flat OpenHypergraph -> DOT/SVG rendering.
// This module implements the missing hierarchical part: DOT clusters whose
// contents are themselves rendered OpenHypergraphs.
pub fn nested_svg(graph: &CompileGraph) -> std::io::Result<Vec<u8>> {
    let mut renderer = NestedDotRenderer::default();
    let dot = renderer.render(graph);
    exec_dot(dot, vec![CommandArg::Format(Format::Svg)])
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
    fn render(&mut self, graph: &CompileGraph) -> String {
        let mut dot = String::new();
        dot.push_str("digraph G {\n");
        dot.push_str("  graph [rankdir=TB, bgcolor=\"#4a4a4a\", compound=true];\n");
        dot.push_str("  node [fontcolor=\"white\", color=\"white\"];\n");
        dot.push_str("  edge [fontcolor=\"white\", color=\"white\"];\n");
        self.render_cluster(graph, &qualified_name(graph), None, &mut dot);
        dot.push_str("}\n");
        dot
    }

    fn render_cluster(
        &mut self,
        graph: &CompileGraph,
        label: &str,
        parent_interface: Option<ParentInterface>,
        dot: &mut String,
    ) -> RenderedInterface {
        let graph_id = self.next_id();
        let prefix = format!("g{graph_id}");
        let cluster_id = cluster_id(graph_id);
        let children = graph
            .children
            .iter()
            .map(|child| (child.operation.as_str(), &child.graph))
            .collect::<HashMap<_, _>>();

        dot.push_str(&format!("  subgraph {cluster_id} {{\n"));
        dot.push_str(&format!("    label=\"{}\";\n", escape_dot_string(label)));
        dot.push_str("    color=\"white\";\n");
        dot.push_str("    fontcolor=\"white\";\n");
        dot.push_str("    style=\"rounded\";\n");

        if let Some(interface) = parent_interface.as_ref() {
            self.render_external_interface(&prefix, interface, dot);
        }
        self.render_nodes(&prefix, &graph.graph, dot);
        self.render_boundary(&prefix, &graph.graph, dot);

        for edge_index in 0..graph.graph.hypergraph.edges.len() {
            let operation = &graph.graph.hypergraph.edges[edge_index];
            if let Some(child) = children.get(operation.to_string().as_str()) {
                let hyperedge = &graph.graph.hypergraph.adjacency[edge_index];
                let child_interface = self.render_cluster(
                    child,
                    &operation.to_string(),
                    Some(ParentInterface {
                        source_labels: hyperedge
                            .sources
                            .iter()
                            .map(|node| graph.graph.hypergraph.nodes[node.0].clone())
                            .collect(),
                        target_labels: hyperedge
                            .targets
                            .iter()
                            .map(|node| graph.graph.hypergraph.nodes[node.0].clone())
                            .collect(),
                    }),
                    dot,
                );
                self.render_nested_connections(
                    &prefix,
                    &graph.graph,
                    edge_index,
                    &child_interface,
                    dot,
                );
            } else {
                self.render_edge_box(&prefix, &graph.graph, edge_index, operation, dot);
            }
        }

        dot.push_str("  }\n");

        if let Some(interface) = parent_interface {
            RenderedInterface {
                cluster_id,
                sources: (0..interface.source_labels.len())
                    .map(|index| interface_source_id(&prefix, index))
                    .collect(),
                targets: (0..interface.target_labels.len())
                    .map(|index| interface_target_id(&prefix, index))
                    .collect(),
            }
        } else {
            RenderedInterface {
                cluster_id,
                sources: graph
                    .graph
                    .sources
                    .iter()
                    .map(|node| node_id(&prefix, *node))
                    .collect(),
                targets: graph
                    .graph
                    .targets
                    .iter()
                    .map(|node| node_id(&prefix, *node))
                    .collect(),
            }
        }
    }

    fn render_external_interface(
        &self,
        prefix: &str,
        interface: &ParentInterface,
        dot: &mut String,
    ) {
        for (index, label) in interface.source_labels.iter().enumerate() {
            self.render_invisible_interface_node(&interface_source_id(prefix, index), dot);
            self.render_interface_label(&interface_source_label_id(prefix, index), label, dot);
            self.render_invisible_alignment(
                &interface_source_label_id(prefix, index),
                &interface_source_id(prefix, index),
                dot,
            );
        }
        for (index, label) in interface.target_labels.iter().enumerate() {
            self.render_invisible_interface_node(&interface_target_id(prefix, index), dot);
            self.render_interface_label(&interface_target_label_id(prefix, index), label, dot);
            self.render_invisible_alignment(
                &interface_target_id(prefix, index),
                &interface_target_label_id(prefix, index),
                dot,
            );
        }
    }

    fn render_interface_label(&self, id: &str, label: &str, dot: &mut String) {
        dot.push_str(&format!(
            "    {id} [shape=plaintext, label=\"{}\"];\n",
            escape_dot_string(label)
        ));
    }

    fn render_invisible_alignment(&self, from: &str, to: &str, dot: &mut String) {
        dot.push_str(&format!("    {from} -> {to} [style=invis, weight=10];\n"));
    }

    fn render_invisible_interface_node(&self, id: &str, dot: &mut String) {
        dot.push_str(&format!(
            "    {id} [shape=point, style=invis, label=\"\", width=0.01, height=0.01];\n"
        ));
    }

    fn render_nodes(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<String, OperationKey>,
        dot: &mut String,
    ) {
        for node_index in 0..graph.hypergraph.nodes.len() {
            let node = NodeId(node_index);
            let label = graph.hypergraph.nodes[node_index].clone();
            dot.push_str(&format!(
                "    {} [shape=point, xlabel=\"{}\"];\n",
                node_id(prefix, node),
                escape_dot_string(&label)
            ));
        }
    }

    fn render_boundary(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<String, OperationKey>,
        dot: &mut String,
    ) {
        for (index, source) in graph.sources.iter().enumerate() {
            dot.push_str(&format!(
                "    {} [shape=point, label=\"\", width=0.05, height=0.05];\n",
                input_id(prefix, index)
            ));
            dot.push_str(&format!(
                "    {} -> {} [style=dashed, dir=none];\n",
                input_id(prefix, index),
                node_id(prefix, *source)
            ));
        }

        for (index, target) in graph.targets.iter().enumerate() {
            dot.push_str(&format!(
                "    {} [shape=point, label=\"\", width=0.05, height=0.05];\n",
                output_id(prefix, index)
            ));
            dot.push_str(&format!(
                "    {} -> {} [style=dashed, dir=none];\n",
                node_id(prefix, *target),
                output_id(prefix, index)
            ));
        }

        if !graph.sources.is_empty() {
            dot.push_str("    { rank=source;");
            for index in 0..graph.sources.len() {
                dot.push_str(&format!(" {}", input_id(prefix, index)));
            }
            dot.push_str(" }\n");
        }

        if !graph.targets.is_empty() {
            dot.push_str("    { rank=sink;");
            for index in 0..graph.targets.len() {
                dot.push_str(&format!(" {}", output_id(prefix, index)));
            }
            dot.push_str(" }\n");
        }
    }

    fn render_edge_box(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<String, OperationKey>,
        edge_index: usize,
        operation: &OperationKey,
        dot: &mut String,
    ) {
        let edge_id = edge_id(prefix, edge_index);
        let hyperedge = &graph.hypergraph.adjacency[edge_index];
        let label = record_label(operation, hyperedge.sources.len(), hyperedge.targets.len());

        dot.push_str(&format!(
            "    {edge_id} [label=\"{}\", shape=record];\n",
            escape_record_label(&label)
        ));

        for (source_index, source) in hyperedge.sources.iter().enumerate() {
            dot.push_str(&format!(
                "    {} -> {edge_id}:s_{source_index};\n",
                node_id(prefix, *source)
            ));
        }
        for (target_index, target) in hyperedge.targets.iter().enumerate() {
            dot.push_str(&format!(
                "    {edge_id}:t_{target_index} -> {};\n",
                node_id(prefix, *target)
            ));
        }
    }

    fn render_nested_connections(
        &self,
        prefix: &str,
        graph: &OpenHypergraph<String, OperationKey>,
        edge_index: usize,
        child: &RenderedInterface,
        dot: &mut String,
    ) {
        let hyperedge = &graph.hypergraph.adjacency[edge_index];
        for (source, child_source) in hyperedge.sources.iter().zip(&child.sources) {
            dot.push_str(&format!(
                "    {} -> {child_source} [lhead={}];\n",
                node_id(prefix, *source),
                child.cluster_id
            ));
        }
        for (child_target, target) in child.targets.iter().zip(&hyperedge.targets) {
            dot.push_str(&format!(
                "    {child_target} -> {} [ltail={}];\n",
                node_id(prefix, *target),
                child.cluster_id
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
    let prefix = match graph.theory {
        catena::compile::GraphTheory::Data => "data",
        catena::compile::GraphTheory::Control => "control",
    };
    format!("{prefix}.{}", graph.definition)
}

fn node_id(prefix: &str, node: NodeId) -> String {
    format!("{prefix}_n_{}", node.0)
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

fn interface_source_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_source_{index}")
}

fn interface_target_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_target_{index}")
}

fn interface_source_label_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_source_label_{index}")
}

fn interface_target_label_id(prefix: &str, index: usize) -> String {
    format!("{prefix}_interface_target_label_{index}")
}

fn record_label(operation: &OperationKey, source_arity: usize, target_arity: usize) -> String {
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

fn escape_dot_string(label: &str) -> String {
    label
        .chars()
        .flat_map(|character| match character {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            _ => vec![character],
        })
        .collect()
}

fn escape_record_label(label: &str) -> String {
    label
        .chars()
        .flat_map(|character| match character {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            _ => vec![character],
        })
        .collect()
}
