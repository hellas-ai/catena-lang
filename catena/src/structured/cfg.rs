use crate::compile::{CompileGraph, CompileTheory};
use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

pub type CfgNodeId = usize;
pub type OperationName = String;
pub type VariableId = usize;
pub type VariableName = String;

pub trait ArrowSemantics {
    fn block_instruction(&self, arrow: &ArrowInstance) -> Option<BlockInstruction> {
        Some(BlockInstruction {
            lhs: arrow.outputs.clone(),
            rhs: BlockInstructionRhs::Primitive {
                operation: arrow.op.clone(),
                args: arrow.inputs.clone(),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowInstance {
    pub id: CfgNodeId,
    pub op: OperationName,
    pub inputs: Vec<VariableId>,
    pub outputs: Vec<VariableId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgEdge {
    pub target: CfgNodeId,
    pub args: Vec<VariableId>,
}

pub struct Region<'a> {
    compile_graph: &'a CompileGraph,
}

impl<'a> Region<'a> {
    pub fn new(
        compile_graph: &'a CompileGraph,
        _node_names: HashMap<NodeId, VariableName>,
    ) -> Self {
        Self { compile_graph }
    }

    fn theory(&self) -> &CompileTheory {
        &self.compile_graph.theory
    }

    fn variable(&self, node: NodeId) -> VariableId {
        node.0
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
    pub(super) params: Vec<VariableId>,
    pub(super) block: Vec<BlockInstruction>,
    pub(super) transfer: Transfer,
}

#[derive(Debug, Clone)]
pub struct BlockInstruction {
    pub lhs: Vec<VariableId>,
    pub rhs: BlockInstructionRhs,
}

#[derive(Debug, Clone)]
pub enum BlockInstructionRhs {
    Primitive {
        operation: OperationName,
        args: Vec<VariableId>,
    },
    Call {
        function: OperationName,
        args: Vec<VariableId>,
    },
}

#[derive(Debug, Clone)]
pub(super) enum Transfer {
    Goto(CfgEdge),
    If {
        condition: VariableId,
        then_edge: CfgEdge,
        else_edge: CfgEdge,
    },
    Switch {
        selector: VariableId,
        edges: Vec<CfgEdge>,
    },
    Return(Vec<VariableId>),
}

impl Cfg {
    pub fn from_region(
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
    ) -> Result<Self, StructuredError> {
        match region.theory() {
            CompileTheory::Data => lower_data_region(region, semantics),
            CompileTheory::Control => lower_control_region(region, semantics),
        }
    }

    pub(super) fn label(&self, node: CfgNodeId) -> String {
        format!("n{node}")
    }
}

// Data regions are lowered as maximal data blocks interrupted by control arrows.
// The instructions in a data block describe dependencies, not execution order.
fn lower_data_region(
    region: &Region<'_>,
    semantics: &impl ArrowSemantics,
) -> Result<Cfg, StructuredError> {
    let mut builder = CfgBuilder::new();
    let entry = builder.new_data_block(variables(region, &source_nodes(region)));
    let mut current_data_block = entry;

    for edge_index in 0..region.operations().len() {
        let arrow = arrow_for_edge(region, edge_index);
        if is_control_operation(region, &arrow.op) {
            current_data_block =
                builder.split_to_control_arrow(region, semantics, current_data_block, &arrow);
        } else {
            builder.append_data_arrow(region, semantics, current_data_block, &arrow);
        }
    }

    builder.return_from(current_data_block, variables(region, &target_nodes(region)));
    Ok(builder.finish(entry))
}

// Control regions are already ordered by control wires. Each control arrow gets
// a CFG node; its transfer is computed from the graph's outgoing control wires.
fn lower_control_region(
    region: &Region<'_>,
    semantics: &impl ArrowSemantics,
) -> Result<Cfg, StructuredError> {
    if region.operations().is_empty() {
        return Err(StructuredError::MissingEntry);
    }

    let mut builder = CfgBuilder::new();
    let arrows = (0..region.operations().len())
        .map(|edge_index| arrow_for_edge(region, edge_index))
        .collect::<Vec<_>>();

    builder.create_control_nodes(region, semantics, &arrows);

    let exit = builder.new_return_node(variables(region, &target_nodes(region)));
    builder.wire_control_transfers(region, &arrows, exit);

    Ok(builder.finish(entry_edge(region)?))
}

struct CfgBuilder {
    nodes: Vec<CfgNode>,
}

impl CfgBuilder {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push_node(&mut self, node: CfgNode) -> CfgNodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    fn new_data_block(&mut self, params: Vec<VariableId>) -> CfgNodeId {
        self.push_node(CfgNode {
            params,
            block: Vec::new(),
            transfer: Transfer::Return(Vec::new()),
        })
    }

    fn new_return_node(&mut self, returns: Vec<VariableId>) -> CfgNodeId {
        self.push_node(CfgNode {
            params: returns.clone(),
            block: Vec::new(),
            transfer: Transfer::Return(returns),
        })
    }

    fn append_data_arrow(
        &mut self,
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
        block: CfgNodeId,
        arrow: &ArrowInstance,
    ) {
        self.nodes[block]
            .block
            .extend(instructions_for_arrow(region, &arrow.op, arrow, semantics));
    }

    // A control arrow in a data region closes the current data block, runs one
    // ordered control step, then resumes data lowering in a fresh block.
    fn split_to_control_arrow(
        &mut self,
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
        data_block: CfgNodeId,
        arrow: &ArrowInstance,
    ) -> CfgNodeId {
        let control_node = self.push_node(CfgNode {
            params: arrow.inputs.clone(),
            block: instructions_for_arrow(region, &arrow.op, arrow, semantics),
            transfer: Transfer::Return(Vec::new()),
        });
        self.nodes[data_block].transfer = Transfer::Goto(CfgEdge {
            target: control_node,
            args: arrow.inputs.clone(),
        });

        let continuation = self.new_data_block(arrow.outputs.clone());
        self.nodes[control_node].transfer = transfer_for_arrow(
            arrow,
            vec![CfgEdge {
                target: continuation,
                args: arrow.outputs.clone(),
            }],
        );
        continuation
    }

    fn create_control_nodes(
        &mut self,
        region: &Region<'_>,
        semantics: &impl ArrowSemantics,
        arrows: &[ArrowInstance],
    ) {
        for arrow in arrows {
            self.push_node(CfgNode {
                params: arrow.inputs.clone(),
                block: instructions_for_arrow(region, &arrow.op, arrow, semantics),
                transfer: Transfer::Return(Vec::new()),
            });
        }
    }

    fn wire_control_transfers(
        &mut self,
        region: &Region<'_>,
        arrows: &[ArrowInstance],
        exit: CfgNodeId,
    ) {
        for arrow in arrows {
            let successors = successors_for_edge(region, arrow.id, Some(exit));
            self.nodes[arrow.id].transfer = transfer_for_arrow(arrow, successors);
        }
    }

    fn return_from(&mut self, block: CfgNodeId, returns: Vec<VariableId>) {
        self.nodes[block].transfer = Transfer::Return(returns);
    }

    fn finish(self, entry: CfgNodeId) -> Cfg {
        let predecessors = predecessors(&self.nodes);
        Cfg {
            entry,
            nodes: self.nodes,
            predecessors,
        }
    }
}

impl CfgNode {
    pub(super) fn successors(&self) -> Vec<CfgNodeId> {
        match &self.transfer {
            Transfer::Goto(edge) => vec![edge.target],
            Transfer::If {
                then_edge,
                else_edge,
                ..
            } => vec![then_edge.target, else_edge.target],
            Transfer::Switch { edges, .. } => edges.iter().map(|edge| edge.target).collect(),
            Transfer::Return(_) => Vec::new(),
        }
    }
}

fn instructions_for_arrow(
    region: &Region<'_>,
    op: &str,
    arrow: &ArrowInstance,
    semantics: &impl ArrowSemantics,
) -> Vec<BlockInstruction> {
    if let Some(child) = region.child_for_operation(op) {
        return vec![call_statement(child, arrow)];
    }
    semantics.block_instruction(arrow).into_iter().collect()
}

fn arrow_for_edge(region: &Region<'_>, edge_index: usize) -> ArrowInstance {
    ArrowInstance {
        id: edge_index,
        op: region.operations()[edge_index].to_string(),
        inputs: variables(region, &region.edge_sources(edge_index)),
        outputs: variables(region, &region.edge_targets(edge_index)),
    }
}

fn is_control_operation(region: &Region<'_>, operation: &str) -> bool {
    operation.starts_with("control.")
        || matches!(
            operation,
            "distr" | "distl" | "val.+.elim" | "merge" | "never" | "elim2"
        )
        || region
            .child_for_operation(operation)
            .is_some_and(|child| matches!(child.theory, CompileTheory::Control))
}

fn source_nodes(region: &Region<'_>) -> Vec<NodeId> {
    region.source_nodes().iter().copied().map(NodeId).collect()
}

fn target_nodes(region: &Region<'_>) -> Vec<NodeId> {
    region.target_nodes().iter().copied().map(NodeId).collect()
}

fn entry_edge(region: &Region<'_>) -> Result<CfgNodeId, StructuredError> {
    if region.operations().is_empty() {
        return Err(StructuredError::MissingEntry);
    }
    let Some(source) = region.source_nodes().first().copied() else {
        return Ok(0);
    };
    Ok((0..region.operations().len())
        .find(|edge| region.edge_sources(*edge).contains(&NodeId(source)))
        .unwrap_or(0))
}

fn successors_for_edge(
    region: &Region<'_>,
    edge_index: usize,
    exit: Option<CfgNodeId>,
) -> Vec<CfgEdge> {
    let mut successors = Vec::new();
    for target in region.edge_targets(edge_index) {
        if region.target_nodes().contains(&target.0) {
            if let Some(exit) = exit {
                push_unique_edge(
                    &mut successors,
                    CfgEdge {
                        target: exit,
                        args: variables(region, &[target]),
                    },
                );
            }
            continue;
        }
        for consumer in consumers_of_wire(region, target) {
            push_unique_edge(
                &mut successors,
                CfgEdge {
                    target: consumer,
                    args: variables(region, &[target]),
                },
            );
        }
    }
    successors
}

fn consumers_of_wire(region: &Region<'_>, wire: NodeId) -> Vec<CfgNodeId> {
    (0..region.operations().len())
        .filter(|edge| region.edge_sources(*edge).contains(&wire))
        .collect()
}

fn call_statement(child: &CompileGraph, arrow: &ArrowInstance) -> BlockInstruction {
    BlockInstruction {
        lhs: arrow.outputs.clone(),
        rhs: BlockInstructionRhs::Call {
            function: child.definition_name.clone(),
            args: arrow.inputs.clone(),
        },
    }
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

fn transfer_for_arrow(arrow: &ArrowInstance, successors: Vec<CfgEdge>) -> Transfer {
    if is_sum_value_elim(&arrow.op) {
        return branch_transfer_for_arrow(arrow, successors);
    }

    match successors.as_slice() {
        [] => Transfer::Return(arrow.outputs.clone()),
        [edge, ..] => Transfer::Goto(edge.clone()),
    }
}

fn branch_transfer_for_arrow(arrow: &ArrowInstance, successors: Vec<CfgEdge>) -> Transfer {
    let selector = arrow.inputs.first().copied().unwrap_or(0);
    match successors.as_slice() {
        [] => Transfer::Return(arrow.outputs.clone()),
        [then_edge, else_edge] => Transfer::If {
            condition: selector,
            then_edge: then_edge.clone(),
            else_edge: else_edge.clone(),
        },
        edges => Transfer::Switch {
            selector,
            edges: edges.to_vec(),
        },
    }
}

fn predecessors(nodes: &[CfgNode]) -> Vec<Vec<CfgNodeId>> {
    let mut predecessors = vec![Vec::new(); nodes.len()];
    for (node_index, node) in nodes.iter().enumerate() {
        for successor in node.successors() {
            predecessors[successor].push(node_index);
        }
    }
    predecessors
}

fn variables(region: &Region<'_>, nodes: &[NodeId]) -> Vec<VariableId> {
    nodes.iter().map(|node| region.variable(*node)).collect()
}

fn push_unique_edge(target: &mut Vec<CfgEdge>, edge: CfgEdge) {
    if !target
        .iter()
        .any(|existing| existing.target == edge.target && existing.args == edge.args)
    {
        target.push(edge);
    }
}

fn is_sum_value_elim(operation: &str) -> bool {
    matches!(operation, "val.+.elim" | "control.val.+.elim")
}

pub(crate) fn variable_name(id: VariableId) -> String {
    if id > usize::MAX / 2 {
        format!("s{}", usize::MAX - id)
    } else {
        format!("w{id}")
    }
}
