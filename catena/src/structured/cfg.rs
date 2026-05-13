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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowInstance {
    pub id: CfgNodeId,
    pub op: OperationName,
    pub inputs: Vec<Variable>,
    pub outputs: Vec<Variable>,
    pub branch_arity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BranchValue {
    Opaque,
    Coproduct(Variable),
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
        let mut branches = Vec::new();
        for (edge_index, op) in f.hypergraph.edges.iter().enumerate() {
            let op = op.to_string();
            let successors = edge_successors(f, edge_index, &consumers, &graph_targets, exit_node);
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
            let (statements, branch) =
                statements_for_arrow(context.child_for_operation(&op), &arrow, semantics);
            branches.push((arrow, branch));
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

        for edge_index in 0..f.hypergraph.edges.len() {
            let (arrow, branch) = branches[edge_index].clone();
            let successors = edge_successors(
                f,
                edge_index,
                &consumers,
                &graph_targets,
                (!graph_targets.is_empty()).then_some(f.hypergraph.edges.len()),
            );
            nodes[edge_index].transfer =
                transfer_for_successors(&mut nodes, arrow, branch, successors, semantics);
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

fn statements_for_arrow(
    child: Option<&CompileGraph>,
    arrow: &ArrowInstance,
    semantics: &impl ArrowSemantics,
) -> (Vec<Stmt>, BranchValue) {
    if let Some(child) = child {
        return (
            statements_for_child_graph(child, arrow),
            branch_value_for_child_graph(child, arrow),
        );
    }
    (semantics.statements(arrow), BranchValue::Opaque)
}

fn statements_for_child_graph(child: &CompileGraph, arrow: &ArrowInstance) -> Vec<Stmt> {
    let mut variables = child_graph_variables(child, arrow);
    (0..child.graph.h.x.0.len())
        .map(|edge_index| {
            let inputs = edge_sources(child, edge_index)
                .into_iter()
                .map(|node| variable_for_child_node(&mut variables, arrow, node))
                .collect();
            let outputs = edge_targets(child, edge_index)
                .into_iter()
                .map(|node| variable_for_child_node(&mut variables, arrow, node))
                .collect();
            Stmt::Primitive(super::ir::Primitive {
                name: child.graph.h.x.0[edge_index].to_string(),
                inputs,
                outputs,
                code: String::new(),
            })
        })
        .collect()
}

fn branch_value_for_child_graph(child: &CompileGraph, arrow: &ArrowInstance) -> BranchValue {
    if arrow.branch_arity <= 1 {
        return BranchValue::Opaque;
    }
    let Some(target) = child.graph.t.table.first() else {
        return BranchValue::Opaque;
    };
    let mut variables = child_graph_variables(child, arrow);
    BranchValue::Coproduct(variable_for_child_node(&mut variables, arrow, *target))
}

fn child_graph_variables(child: &CompileGraph, arrow: &ArrowInstance) -> HashMap<usize, Variable> {
    let mut variables = HashMap::new();
    for (index, node) in child.graph.s.table.iter().enumerate() {
        if let Some(input) = arrow.inputs.get(index) {
            variables.insert(*node, input.clone());
        }
    }

    if arrow.branch_arity > 1 && child.graph.t.table.len() == 1 {
        variables.insert(child.graph.t.table[0], branch_result_variable(arrow));
    } else {
        for (index, node) in child.graph.t.table.iter().enumerate() {
            if let Some(output) = arrow.outputs.get(index) {
                variables.insert(*node, output.clone());
            }
        }
    }
    variables
}

fn variable_for_child_node(
    variables: &mut HashMap<usize, Variable>,
    arrow: &ArrowInstance,
    node: usize,
) -> Variable {
    variables
        .entry(node)
        .or_insert_with(|| format!("v{}_{}", arrow.id, node))
        .clone()
}

fn edge_sources(child: &CompileGraph, edge_index: usize) -> Vec<usize> {
    child
        .graph
        .h
        .s
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|sources| sources.table.0)
        .unwrap_or_default()
}

fn edge_targets(child: &CompileGraph, edge_index: usize) -> Vec<usize> {
    child
        .graph
        .h
        .t
        .clone()
        .into_iter()
        .nth(edge_index)
        .map(|targets| targets.table.0)
        .unwrap_or_default()
}

fn edge_successors(
    f: &OpenHypergraph<Obj, Arr>,
    edge_index: CfgNodeId,
    consumers: &HashMap<NodeId, Vec<CfgNodeId>>,
    graph_targets: &HashSet<NodeId>,
    exit_node: Option<CfgNodeId>,
) -> Vec<CfgNodeId> {
    let mut successors = Vec::new();
    for target in &f.hypergraph.adjacency[edge_index].targets {
        if graph_targets.contains(target) {
            if let Some(exit_node) = exit_node {
                push_unique_all(&mut successors, [exit_node]);
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
    branch: BranchValue,
    successors: Vec<CfgNodeId>,
    semantics: &impl ArrowSemantics,
) -> Transfer {
    match successors.as_slice() {
        [] => Transfer::Return,
        [target] => Transfer::Goto(*target),
        [then_target, else_target] => {
            let condition = branch_condition_value(&arrow, 0);
            let payload = branch_payload(&arrow, &branch);
            let then_target =
                append_binding_node(nodes, branch_binding(&arrow, 0, &payload), *then_target);
            let else_target =
                append_binding_node(nodes, branch_binding(&arrow, 1, &payload), *else_target);
            let branch_node = nodes.len();
            nodes.push(CfgNode {
                statements: vec![Stmt::Assign {
                    lhs: condition.clone(),
                    rhs: branch_condition_rhs(&arrow, &branch, 0, semantics),
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
            let payload = branch_payload(&arrow, &branch);
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
                    selector: branch_selector(&arrow, &branch, semantics),
                    targets,
                },
            });
            Transfer::Goto(branch_node)
        }
    }
}

fn branch_condition_rhs(
    arrow: &ArrowInstance,
    branch: &BranchValue,
    output: usize,
    semantics: &impl ArrowSemantics,
) -> Expr {
    match branch {
        BranchValue::Opaque => semantics.branch_condition_rhs(arrow, output),
        BranchValue::Coproduct(value) => format!("{value}.tag == {output}"),
    }
}

fn branch_selector(
    arrow: &ArrowInstance,
    branch: &BranchValue,
    semantics: &impl ArrowSemantics,
) -> Variable {
    match branch {
        BranchValue::Opaque => semantics.selector(arrow),
        BranchValue::Coproduct(value) => format!("{value}.tag"),
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

fn branch_payload(arrow: &ArrowInstance, branch: &BranchValue) -> Variable {
    match branch {
        BranchValue::Opaque => format!("p{}", arrow.id),
        BranchValue::Coproduct(value) => format!("{value}.payload"),
    }
}

fn branch_result_variable(arrow: &ArrowInstance) -> Variable {
    format!("r{}", arrow.id)
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
