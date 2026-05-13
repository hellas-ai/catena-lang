use super::ir::Stmt;
use crate::compile::CompileGraph;
use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::{NodeId, OpenHypergraph};
use std::collections::{HashMap, HashSet};

pub type CfgNodeId = usize;
pub type Expr = String;
pub type OperationName = String;
pub type Variable = String;

pub trait ArrowSemantics {
    fn statements(&self, arrow: &ArrowInstance) -> Vec<Stmt>;

    fn branch_condition_rhs(&self, arrow: &ArrowInstance, output: usize) -> Expr {
        format!("/* {} output {output} */ 1", sanitize_ident(&arrow.op))
    }

    fn selector(&self, arrow: &ArrowInstance) -> Variable {
        format!("/* {} */ 0", sanitize_ident(&arrow.op))
    }

    fn counted_loop(&self, _op: &str) -> Option<(Variable, Expr)> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowInstance {
    pub id: CfgNodeId,
    pub op: OperationName,
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub branch_arity: usize,
}

pub struct Context<'a> {
    graph: &'a CompileGraph,
}

impl<'a> Context<'a> {
    pub fn new(graph: &'a CompileGraph) -> Self {
        Self { graph }
    }

    pub fn child_for_operation(&self, operation: &str) -> Option<&'a CompileGraph> {
        self.graph
            .children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.graph)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    #[error("shallow graph has no operation reachable from the source interface")]
    MissingEntry,
    #[error("control-flow graph has an irreducible back edge from {from} to {to}")]
    IrreducibleBackEdge { from: String, to: String },
    #[error("branch target {0} is not in the structured context")]
    MissingContext(String),
    #[error("control node {node} has {successors} entry successors; only one entry is supported")]
    UnsupportedEntry { node: String, successors: usize },
}

#[derive(Debug, Clone)]
pub struct Cfg {
    pub(super) entry: CfgNodeId,
    pub(super) nodes: Vec<CfgNode>,
    pub(super) predecessors: Vec<Vec<CfgNodeId>>,
}

#[derive(Debug, Clone)]
pub(super) struct CfgNode {
    pub(super) op: OperationName,
    pub(super) statements: Vec<Stmt>,
    pub(super) transfer: Transfer,
    pub(super) counted_loop: Option<(Variable, Expr)>,
}

#[derive(Debug, Clone)]
pub(super) enum Transfer {
    Goto(CfgNodeId),
    If {
        condition: Variable,
        then_target: CfgNodeId,
        else_target: CfgNodeId,
    },
    Switch {
        selector: Variable,
        targets: Vec<CfgNodeId>,
    },
    Return,
}

impl Cfg {
    pub fn from_hypergraph(
        f: &OpenHypergraph<Obj, Arr>,
        context: &Context<'_>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let mut consumers: HashMap<NodeId, Vec<CfgNodeId>> = HashMap::new();
        for (edge_index, adjacency) in f.hypergraph.adjacency.iter().enumerate() {
            for source in &adjacency.sources {
                consumers.entry(*source).or_default().push(edge_index);
            }
        }

        let mut entry_edges = Vec::new();
        // One structured program has one entry point. Additional open sources
        // are external state alternatives, not extra CFG entries.
        if let Some(source) = f.sources.first() {
            if let Some(edges) = consumers.get(source) {
                push_unique_all(&mut entry_edges, edges.iter().copied());
            }
        }
        if entry_edges.is_empty() && !f.hypergraph.edges.is_empty() {
            entry_edges.push(0);
        }

        let entry = match entry_edges.as_slice() {
            [edge] => *edge,
            [] => return Err(StructuredError::MissingEntry),
            _ => {
                return Err(StructuredError::UnsupportedEntry {
                    node: "entry".to_string(),
                    successors: entry_edges.len(),
                });
            }
        };

        let graph_targets: HashSet<NodeId> = f.targets.iter().copied().collect();
        let exit_node = (!graph_targets.is_empty()).then_some(f.hypergraph.edges.len());
        let mut nodes = Vec::new();
        let mut arrows = Vec::new();
        for (edge_index, op) in f.hypergraph.edges.iter().enumerate() {
            let op = op.to_string();
            let _nested = context.child_for_operation(&op);
            let successors =
                edge_successors(f, edge_index, &consumers, &graph_targets, exit_node, &op);
            let arrow = ArrowInstance {
                id: edge_index,
                op: op.clone(),
                inputs: f.hypergraph.adjacency[edge_index]
                    .sources
                    .iter()
                    .map(|node| wire_name(*node))
                    .collect(),
                outputs: f.hypergraph.adjacency[edge_index]
                    .targets
                    .iter()
                    .map(|node| wire_name(*node))
                    .collect(),
                branch_arity: successors.len(),
            };
            arrows.push(arrow.clone());
            nodes.push(CfgNode {
                counted_loop: semantics.counted_loop(&op),
                op,
                statements: semantics.statements(&arrow),
                transfer: Transfer::Return,
            });
        }

        if !graph_targets.is_empty() {
            nodes.push(CfgNode {
                op: "__exit".to_string(),
                statements: Vec::new(),
                transfer: Transfer::Return,
                counted_loop: None,
            });
        }

        for edge_index in 0..f.hypergraph.edges.len() {
            let arrow = arrows[edge_index].clone();
            let successors = edge_successors(
                f,
                edge_index,
                &consumers,
                &graph_targets,
                (!graph_targets.is_empty()).then_some(f.hypergraph.edges.len()),
                &nodes[edge_index].op,
            );
            nodes[edge_index].transfer =
                transfer_for_successors(&mut nodes, arrow, successors, semantics);
        }

        let mut predecessors = vec![Vec::new(); nodes.len()];
        for (node_index, node) in nodes.iter().enumerate() {
            for successor in node.successors() {
                predecessors[successor].push(node_index);
            }
        }

        Ok(Self {
            entry,
            nodes,
            predecessors,
        })
    }

    pub(super) fn label(&self, node: CfgNodeId) -> String {
        format!("n{node}")
    }
}

impl CfgNode {
    pub(super) fn successors(&self) -> Vec<CfgNodeId> {
        match &self.transfer {
            Transfer::Goto(target) => vec![*target],
            Transfer::If {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
            Transfer::Switch { targets, .. } => targets.clone(),
            Transfer::Return => Vec::new(),
        }
    }
}

fn edge_successors(
    f: &OpenHypergraph<Obj, Arr>,
    edge_index: CfgNodeId,
    consumers: &HashMap<NodeId, Vec<CfgNodeId>>,
    graph_targets: &HashSet<NodeId>,
    exit_node: Option<CfgNodeId>,
    op: &str,
) -> Vec<CfgNodeId> {
    let mut successors = Vec::new();
    for target in &f.hypergraph.adjacency[edge_index].targets {
        if graph_targets.contains(target) {
            if boundary_target_is_control_successor(op) {
                if let Some(exit_node) = exit_node {
                    push_unique_all(&mut successors, [exit_node]);
                }
            }
            continue;
        }
        if let Some(edges) = consumers.get(target) {
            push_unique_all(&mut successors, edges.iter().copied());
        }
    }
    successors
}

fn transfer_for_successors(
    nodes: &mut Vec<CfgNode>,
    arrow: ArrowInstance,
    successors: Vec<CfgNodeId>,
    semantics: &impl ArrowSemantics,
) -> Transfer {
    match successors.as_slice() {
        [] => Transfer::Return,
        [target] => Transfer::Goto(*target),
        [then_target, else_target] => {
            let condition = branch_condition_value(&arrow, 0);
            let payload = branch_payload(&arrow);
            let then_target = append_binding_node(
                nodes,
                &arrow,
                0,
                branch_binding(&arrow, 0, &payload),
                *then_target,
            );
            let else_target = append_binding_node(
                nodes,
                &arrow,
                1,
                branch_binding(&arrow, 1, &payload),
                *else_target,
            );
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                op: format!("{}.__branch", arrow.op),
                statements: vec![Stmt::Assign {
                    lhs: condition.clone(),
                    rhs: semantics.branch_condition_rhs(&arrow, 0),
                }],
                transfer: Transfer::If {
                    condition,
                    then_target,
                    else_target,
                },
                counted_loop: None,
            });
            Transfer::Goto(branch_node)
        }
        targets => {
            let payload = branch_payload(&arrow);
            let targets = targets
                .iter()
                .enumerate()
                .map(|(index, target)| {
                    append_binding_node(
                        nodes,
                        &arrow,
                        index,
                        branch_binding(&arrow, index, &payload),
                        *target,
                    )
                })
                .collect();
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                op: format!("{}.__branch", arrow.op),
                statements: Vec::new(),
                transfer: Transfer::Switch {
                    selector: semantics.selector(&arrow),
                    targets,
                },
                counted_loop: None,
            });
            Transfer::Goto(branch_node)
        }
    }
}

fn append_binding_node(
    nodes: &mut Vec<CfgNode>,
    arrow: &ArrowInstance,
    output: usize,
    bind: Option<(Variable, Variable)>,
    target: CfgNodeId,
) -> CfgNodeId {
    let Some((lhs, rhs)) = bind else {
        return target;
    };
    let node = nodes.len();
    nodes.push(CfgNode {
        op: format!("{}.__bind{output}", arrow.op),
        statements: vec![Stmt::Assign { lhs, rhs }],
        transfer: Transfer::Goto(target),
        counted_loop: None,
    });
    node
}

fn branch_payload(arrow: &ArrowInstance) -> Variable {
    format!("p{}", arrow.id)
}

fn branch_condition_value(arrow: &ArrowInstance, output: usize) -> Variable {
    format!("c{}_{}", arrow.id, output)
}

fn branch_binding(
    arrow: &ArrowInstance,
    output: usize,
    payload: &str,
) -> Option<(Variable, Variable)> {
    arrow
        .outputs
        .get(output)
        .map(|wire| (wire.clone(), payload.to_string()))
}

fn boundary_target_is_control_successor(op: &str) -> bool {
    // `gpu.sync` exposes phase-token outputs at the graph boundary, but those
    // are data/control products rather than alternate control exits.
    op != "gpu.sync"
}

fn wire_name(node: NodeId) -> Variable {
    format!("w{}", node.0)
}

fn push_unique_all(target: &mut Vec<CfgNodeId>, values: impl IntoIterator<Item = CfgNodeId>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
