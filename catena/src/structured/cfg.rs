use super::ir::Stmt;
use crate::compile::{CompileGraph, CompileTheory};
use open_hypergraphs::lax::NodeId;
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowInstance {
    pub id: CfgNodeId,
    pub op: OperationName,
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub branch_arity: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    Primitive,
    ChildRegion,
}

#[derive(Debug, Clone)]
struct DataNode {
    edge: usize,
    op: OperationName,
    kind: NodeKind,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

pub struct Region<'a> {
    compile_graph: &'a CompileGraph,
    node_names: HashMap<NodeId, Variable>,
}

impl<'a> Region<'a> {
    pub fn new(compile_graph: &'a CompileGraph, node_names: HashMap<NodeId, Variable>) -> Self {
        Self {
            compile_graph,
            node_names,
        }
    }

    fn theory(&self) -> &CompileTheory {
        &self.compile_graph.theory
    }

    fn variable(&self, node: NodeId) -> Variable {
        self.node_names
            .get(&node)
            .cloned()
            .unwrap_or_else(|| format!("w{}", node.0))
    }

    fn child_for_operation(&self, operation: &str) -> Option<&CompileGraph> {
        self.compile_graph
            .children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.graph)
    }

    fn source_nodes(&self) -> &[usize] {
        &self.compile_graph.graph.s.table
    }

    fn target_nodes(&self) -> &[usize] {
        &self.compile_graph.graph.t.table
    }

    fn operations(&self) -> &[crate::lang::Arr] {
        &self.compile_graph.graph.h.x.0
    }

    fn edge_sources(&self, edge_index: usize) -> Vec<NodeId> {
        edge_sources(self, edge_index)
    }

    fn edge_targets(&self, edge_index: usize) -> Vec<NodeId> {
        edge_targets(self, edge_index)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    #[error("shallow graph has no operation reachable from the source interface")]
    MissingEntry,
    #[error("control-flow graph has an irreducible back edge from {from} to {to}")]
    IrreducibleBackEdge { from: String, to: String },
    #[error("dataflow graph has a cycle among remaining operations: {operations}")]
    DataflowCycle { operations: String },
    #[error("dataflow graph depends on unavailable wires: {wires}; remaining: {remaining}")]
    UnavailableWires { wires: String, remaining: String },
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
    pub(super) statements: Vec<Stmt>,
    pub(super) transfer: Transfer,
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
    pub fn from_region(
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        match region.theory() {
            CompileTheory::Data => Self::from_dataflow_region(region, semantics),
            CompileTheory::Control => Self::from_control_region(region, semantics),
        }
    }

    fn from_control_region(
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let mut consumers: HashMap<NodeId, Vec<CfgNodeId>> = HashMap::new();
        for edge_index in 0..region.operations().len() {
            for source in region.edge_sources(edge_index) {
                consumers.entry(source).or_default().push(edge_index);
            }
        }

        let mut entry_edges = Vec::new();
        // One structured program has one entry point. Additional open sources
        // are external state alternatives, not extra CFG entries.
        if let Some(source) = region.source_nodes().first() {
            if let Some(edges) = consumers.get(&NodeId(*source)) {
                push_unique_all(&mut entry_edges, edges.iter().copied());
            }
        }
        if entry_edges.is_empty() && !region.operations().is_empty() {
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

        let graph_targets = region
            .target_nodes()
            .iter()
            .copied()
            .map(NodeId)
            .collect::<HashSet<_>>();
        let exit_node = (!graph_targets.is_empty()).then_some(region.operations().len());
        let mut nodes = Vec::new();
        let mut branches = Vec::new();
        for (edge_index, op) in region.operations().iter().enumerate() {
            let op = op.to_string();
            let successors =
                edge_successors(region, edge_index, &consumers, &graph_targets, exit_node);
            let arrow = ArrowInstance {
                id: edge_index,
                op: op.clone(),
                inputs: region
                    .edge_sources(edge_index)
                    .iter()
                    .map(|node| region.variable(*node))
                    .collect(),
                outputs: region
                    .edge_targets(edge_index)
                    .iter()
                    .map(|node| region.variable(*node))
                    .collect(),
                branch_arity: successors.len(),
            };
            let statements = statements_for_arrow(region, &op, &arrow, semantics);
            branches.push(arrow);
            nodes.push(CfgNode {
                statements,
                transfer: Transfer::Return,
            });
        }

        if !graph_targets.is_empty() {
            nodes.push(CfgNode {
                statements: Vec::new(),
                transfer: Transfer::Return,
            });
        }

        for edge_index in 0..region.operations().len() {
            let arrow = branches[edge_index].clone();
            let successors = edge_successors(
                region,
                edge_index,
                &consumers,
                &graph_targets,
                (!graph_targets.is_empty()).then_some(region.operations().len()),
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

    // Schedule dataflow edges with a topological sort over wire dependencies, then place the resulting SSA-like primitive statements in one CFG node.
    // for now we assume dataflow graphs are acyclic
    // but we may want to relax this condition
    fn from_dataflow_region(
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let data_nodes = data_nodes(region);
        let mut available = region
            .source_nodes()
            .iter()
            .map(|node| NodeId(*node))
            .collect::<HashSet<_>>();
        let mut remaining = (0..data_nodes.len()).collect::<Vec<_>>();
        let mut statements = Vec::new();

        // Data regions are scheduled by wire dependencies. Each scheduled node
        // is either a primitive operation or a child-region call; both expose
        // explicit inputs and outputs to the scheduler.
        while !remaining.is_empty() {
            let Some(index) = remaining.iter().position(|edge| {
                data_nodes[*edge]
                    .inputs
                    .iter()
                    .all(|source| available.contains(source))
            }) else {
                return Err(classify_dataflow_block(
                    region,
                    &data_nodes,
                    &available,
                    &remaining,
                ));
            };

            let node_index = remaining.remove(index);
            let data_node = &data_nodes[node_index];
            let arrow = arrow_for_data_node(data_node, region);
            statements.extend(lower_data_node(region, data_node, &arrow, semantics));
            available.extend(data_node.outputs.iter().copied());
        }

        Ok(Self {
            entry: 0,
            nodes: vec![CfgNode {
                statements,
                transfer: Transfer::Return,
            }],
            predecessors: vec![Vec::new()],
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

fn statements_for_arrow(
    region: &Region<'_>,
    op: &str,
    arrow: &ArrowInstance,
    semantics: &impl ArrowSemantics,
) -> Vec<Stmt> {
    if let Some(child) = region.child_for_operation(op) {
        return vec![call_statement(child, arrow)];
    }
    semantics.statements(arrow)
}

fn call_statement(child: &CompileGraph, arrow: &ArrowInstance) -> Stmt {
    let outputs = if arrow.branch_arity > 1 {
        vec![format!("b{}", arrow.id), format!("p{}", arrow.id)]
    } else {
        arrow.outputs.clone()
    };
    Stmt::Call {
        function: child.definition_name.clone(),
        inputs: arrow.inputs.clone(),
        outputs,
    }
}

fn data_nodes(region: &Region<'_>) -> Vec<DataNode> {
    region
        .operations()
        .iter()
        .enumerate()
        .map(|(edge, op)| {
            let op = op.to_string();
            DataNode {
                edge,
                kind: if region.child_for_operation(&op).is_some() {
                    NodeKind::ChildRegion
                } else {
                    NodeKind::Primitive
                },
                op,
                inputs: region.edge_sources(edge),
                outputs: region.edge_targets(edge),
            }
        })
        .collect()
}

fn arrow_for_data_node(node: &DataNode, region: &Region<'_>) -> ArrowInstance {
    ArrowInstance {
        id: node.edge,
        op: node.op.clone(),
        inputs: node
            .inputs
            .iter()
            .map(|node| region.variable(*node))
            .collect(),
        outputs: node
            .outputs
            .iter()
            .map(|node| region.variable(*node))
            .collect(),
        branch_arity: 0,
    }
}

fn lower_data_node(
    region: &Region<'_>,
    node: &DataNode,
    arrow: &ArrowInstance,
    _semantics: &impl ArrowSemantics,
) -> Vec<Stmt> {
    match node.kind {
        NodeKind::Primitive => vec![Stmt::Primitive(super::ir::Primitive {
            name: node.op.clone(),
            inputs: arrow.inputs.clone(),
            outputs: arrow.outputs.clone(),
            code: String::new(),
        })],
        NodeKind::ChildRegion => {
            let child = region
                .child_for_operation(&node.op)
                .expect("child region node must have child context");
            vec![call_statement(child, arrow)]
        }
    }
}

fn classify_dataflow_block(
    region: &Region<'_>,
    nodes: &[DataNode],
    available: &HashSet<NodeId>,
    remaining: &[usize],
) -> StructuredError {
    let remaining_set = remaining.iter().copied().collect::<HashSet<_>>();
    let remaining_targets = remaining
        .iter()
        .flat_map(|node| nodes[*node].outputs.iter().copied())
        .collect::<HashSet<_>>();

    let unavailable = remaining
        .iter()
        .flat_map(|node| {
            nodes[*node]
                .inputs
                .iter()
                .copied()
                .filter(|source| !available.contains(source) && !remaining_targets.contains(source))
                .map(|source| blocked_wire_description(region, &nodes[*node], source))
        })
        .collect::<Vec<_>>();

    if !unavailable.is_empty() {
        return StructuredError::UnavailableWires {
            wires: unavailable.join(", "),
            remaining: remaining_node_descriptions(region, nodes, remaining).join("; "),
        };
    }

    let operations = remaining_set
        .iter()
        .copied()
        .map(|node| format!("{}#{}", nodes[node].op, nodes[node].edge))
        .collect::<Vec<_>>()
        .join(", ");
    StructuredError::DataflowCycle { operations }
}

fn blocked_wire_description(region: &Region<'_>, node: &DataNode, source: NodeId) -> String {
    format!("{} input {}", node.op, region.variable(source))
}

fn remaining_node_descriptions(
    region: &Region<'_>,
    nodes: &[DataNode],
    remaining: &[usize],
) -> Vec<String> {
    remaining
        .iter()
        .map(|node| node_description(region, &nodes[*node]))
        .collect()
}

fn node_description(region: &Region<'_>, node: &DataNode) -> String {
    let inputs = node
        .inputs
        .iter()
        .map(|node| region.variable(*node))
        .collect::<Vec<_>>()
        .join(",");
    let outputs = node
        .outputs
        .iter()
        .map(|node| region.variable(*node))
        .collect::<Vec<_>>()
        .join(",");
    format!("{}({inputs}) -> ({outputs})", node.op)
}

fn edge_successors(
    region: &Region<'_>,
    edge_index: CfgNodeId,
    consumers: &HashMap<NodeId, Vec<CfgNodeId>>,
    graph_targets: &HashSet<NodeId>,
    exit_node: Option<CfgNodeId>,
) -> Vec<CfgNodeId> {
    let mut successors = Vec::new();
    for target in region.edge_targets(edge_index) {
        if graph_targets.contains(&target) {
            if let Some(exit_node) = exit_node {
                push_unique_all(&mut successors, [exit_node]);
            }
            continue;
        }
        if let Some(edges) = consumers.get(&target) {
            push_unique_all(&mut successors, edges.iter().copied());
        }
    }
    successors
}

fn edge_sources(region: &Region<'_>, edge_index: usize) -> Vec<NodeId> {
    region
        .compile_graph
        .graph
        .h
        .s
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|sources| sources.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
}

fn edge_targets(region: &Region<'_>, edge_index: usize) -> Vec<NodeId> {
    region
        .compile_graph
        .graph
        .h
        .t
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|targets| targets.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
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
            let then_target =
                append_binding_node(nodes, branch_binding(&arrow, 0, &payload), *then_target);
            let else_target =
                append_binding_node(nodes, branch_binding(&arrow, 1, &payload), *else_target);
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                statements: vec![Stmt::Assign {
                    lhs: condition.clone(),
                    rhs: semantics.branch_condition_rhs(&arrow, 0),
                }],
                transfer: Transfer::If {
                    condition,
                    then_target,
                    else_target,
                },
            });
            Transfer::Goto(branch_node)
        }
        targets => {
            let payload = branch_payload(&arrow);
            let targets = targets
                .iter()
                .enumerate()
                .map(|(index, target)| {
                    append_binding_node(nodes, branch_binding(&arrow, index, &payload), *target)
                })
                .collect();
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                statements: Vec::new(),
                transfer: Transfer::Switch {
                    selector: semantics.selector(&arrow),
                    targets,
                },
            });
            Transfer::Goto(branch_node)
        }
    }
}

fn append_binding_node(
    nodes: &mut Vec<CfgNode>,
    bind: Option<(Variable, Variable)>,
    target: CfgNodeId,
) -> CfgNodeId {
    let Some((lhs, rhs)) = bind else {
        return target;
    };
    let node = nodes.len();
    nodes.push(CfgNode {
        statements: vec![Stmt::Assign { lhs, rhs }],
        transfer: Transfer::Goto(target),
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
