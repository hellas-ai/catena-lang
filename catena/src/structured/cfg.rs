use super::ir::Stmt;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphKind {
    Data,
    Control,
}

#[derive(Debug, Clone)]
struct ChildContext {
    pub operation: String,
    pub context: BuildContext,
}

#[derive(Debug, Clone)]
pub struct BuildContext {
    kind: GraphKind,
    graph: OpenHypergraph<Obj, Arr>,
    children: Vec<ChildContext>,
    variables: HashMap<NodeId, Variable>,
    prefix: String,
}

impl BuildContext {
    pub fn new(
        kind: GraphKind,
        graph: OpenHypergraph<Obj, Arr>,
        children: Vec<(String, BuildContext)>,
    ) -> Self {
        Self {
            kind,
            graph,
            children: children
                .into_iter()
                .map(|(operation, context)| ChildContext { operation, context })
                .collect(),
            variables: HashMap::new(),
            prefix: "w".to_string(),
        }
    }

    pub fn with_variables(mut self, variables: HashMap<NodeId, Variable>) -> Self {
        self.variables = variables;
        self
    }

    pub fn graph(&self) -> &OpenHypergraph<Obj, Arr> {
        &self.graph
    }

    pub fn kind(&self) -> GraphKind {
        self.kind
    }

    fn variable(&self, node: NodeId) -> Variable {
        self.variables
            .get(&node)
            .cloned()
            .unwrap_or_else(|| format!("{}{}", self.prefix, node.0))
    }

    fn child_for_operation(&self, operation: &str) -> Option<&BuildContext> {
        self.children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.context)
    }

    fn with_child_variables(&self, variables: HashMap<NodeId, Variable>, prefix: String) -> Self {
        Self {
            kind: self.kind,
            graph: self.graph.clone(),
            children: self.children.clone(),
            variables,
            prefix,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    #[error("shallow graph has no operation reachable from the source interface")]
    MissingEntry,
    #[error("control-flow graph has an irreducible back edge from {from} to {to}")]
    IrreducibleBackEdge { from: String, to: String },
    #[error("dataflow graph has a cycle or depends on unavailable wires")]
    DataflowCycle,
    #[error("branch target {0} is not in the structured context")]
    MissingContext(String),
    #[error(
        "expected alternating structured graph layers, but {parent:?} graph contains {child:?} child"
    )]
    InvalidLayer { parent: GraphKind, child: GraphKind },
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
    pub fn from_control_context(
        context: &BuildContext,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let f = context.graph();
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
                    .map(|node| context.variable(*node))
                    .collect(),
                outputs: f.hypergraph.adjacency[edge_index]
                    .targets
                    .iter()
                    .map(|node| context.variable(*node))
                    .collect(),
                branch_arity: successors.len(),
            };
            let (statements, branch) = statements_for_arrow(context, &op, &arrow, semantics)?;
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

    // Schedule dataflow edges with a topological sort over wire dependencies, then place the resulting SSA-like primitive statements in one CFG node.
    // for now we assume dataflow graphs are acyclic
    // but we may want to relax this condition
    pub fn from_dataflow_context(
        context: &BuildContext,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        let statements = dataflow_statements(context, semantics)?;
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
    context: &BuildContext,
    op: &str,
    arrow: &ArrowInstance,
    semantics: &impl ArrowSemantics,
) -> Result<(Vec<Stmt>, BranchValue), StructuredError> {
    if let Some(child) = context.child_for_operation(op) {
        return Ok((
            statements_for_child_graph(context.kind(), child, arrow, semantics)?,
            branch_value_for_child_graph(child, arrow),
        ));
    }
    Ok((semantics.statements(arrow), BranchValue::Opaque))
}

fn statements_for_child_graph(
    parent: GraphKind,
    child: &BuildContext,
    arrow: &ArrowInstance,
    semantics: &impl ArrowSemantics,
) -> Result<Vec<Stmt>, StructuredError> {
    if child.kind() == parent {
        return Err(StructuredError::InvalidLayer {
            parent,
            child: child.kind(),
        });
    }

    let child_context = child.with_child_variables(
        child_graph_variables(child, arrow),
        format!("v{}_", arrow.id),
    );
    match child.kind() {
        GraphKind::Data => dataflow_statements(&child_context, semantics),
        GraphKind::Control => {
            super::ramsey::structure(Cfg::from_control_context(&child_context, semantics)?)
        }
    }
}

fn dataflow_statements(
    context: &BuildContext,
    semantics: &impl ArrowSemantics,
) -> Result<Vec<Stmt>, StructuredError> {
    let f = context.graph();
    let mut available = f.sources.iter().copied().collect::<HashSet<_>>();
    let mut remaining = (0..f.hypergraph.edges.len()).collect::<Vec<_>>();
    let mut variables = context.variables.clone();
    let mut statements = Vec::new();

    // For now structured dataflow assumes an acyclic dependency graph.
    while !remaining.is_empty() {
        let Some(index) = remaining.iter().position(|edge| {
            f.hypergraph.adjacency[*edge]
                .sources
                .iter()
                .all(|source| available.contains(source))
        }) else {
            return Err(StructuredError::DataflowCycle);
        };

        let edge_index = remaining.remove(index);
        let op = f.hypergraph.edges[edge_index].to_string();
        let adjacency = &f.hypergraph.adjacency[edge_index];
        let arrow = ArrowInstance {
            id: edge_index,
            op: op.clone(),
            inputs: adjacency
                .sources
                .iter()
                .map(|node| variable_for_child_node(&mut variables, &context.prefix, *node))
                .collect(),
            outputs: adjacency
                .targets
                .iter()
                .map(|node| variable_for_child_node(&mut variables, &context.prefix, *node))
                .collect(),
            branch_arity: 0,
        };

        if let Some(child) = context.child_for_operation(&op) {
            statements.extend(statements_for_child_graph(
                context.kind(),
                child,
                &arrow,
                semantics,
            )?);
        } else {
            statements.push(Stmt::Primitive(super::ir::Primitive {
                name: op,
                inputs: arrow.inputs,
                outputs: arrow.outputs,
                code: String::new(),
            }));
        }
        available.extend(adjacency.targets.iter().copied());
    }

    Ok(statements)
}

fn branch_value_for_child_graph(child: &BuildContext, arrow: &ArrowInstance) -> BranchValue {
    if arrow.branch_arity <= 1 {
        return BranchValue::Opaque;
    }
    let Some(target) = child.graph().targets.first() else {
        return BranchValue::Opaque;
    };
    let mut variables = child_graph_variables(child, arrow);
    BranchValue::Coproduct(variable_for_child_node(
        &mut variables,
        &format!("v{}_", arrow.id),
        *target,
    ))
}

fn child_graph_variables(child: &BuildContext, arrow: &ArrowInstance) -> HashMap<NodeId, Variable> {
    let mut variables = HashMap::new();
    for (index, node) in child.graph().sources.iter().enumerate() {
        if let Some(input) = arrow.inputs.get(index) {
            variables.insert(*node, input.clone());
        }
    }

    if arrow.branch_arity > 1 && child.graph().targets.len() == 1 {
        variables.insert(child.graph().targets[0], branch_result_variable(arrow));
    } else {
        for (index, node) in child.graph().targets.iter().enumerate() {
            if let Some(output) = arrow.outputs.get(index) {
                variables.insert(*node, output.clone());
            }
        }
    }
    variables
}

fn variable_for_child_node(
    variables: &mut HashMap<NodeId, Variable>,
    prefix: &str,
    node: NodeId,
) -> Variable {
    variables
        .entry(node)
        .or_insert_with(|| format!("{prefix}{}", node.0))
        .clone()
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
