use crate::compile::{CompileGraph, CompileTheory};
use open_hypergraphs::lax::NodeId;
use std::collections::{HashMap, HashSet};

pub type CfgNodeId = usize;
pub type OperationId = usize;
pub type OperationName = String;
pub type VariableId = usize;
pub type VariableName = String;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationInstance {
    pub id: OperationId,
    pub name: OperationName,
    pub inputs: Vec<VariableId>,
    pub outputs: Vec<VariableId>,
}

#[derive(Debug, thiserror::Error)]
pub enum CfgError {
    #[error("cfg only accepts data regions; got {0}")]
    UnsupportedTheory(CompileTheory),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Cfg {
    pub(super) entry: CfgNodeId,
    pub(super) nodes: Vec<CfgNode>,
    pub(super) predecessors: Vec<Vec<CfgNodeId>>,
}

#[derive(Debug, Clone)]
pub struct CfgNode {
    pub id: CfgNodeId,
    pub params: Vec<VariableId>,
    pub block: Vec<BlockInstruction>,
    pub transfer: Transfer,
}

#[derive(Debug, Clone)]
pub struct BlockInstruction {
    pub operation_id: OperationId,
    pub operation: OperationName,
    pub args: Vec<VariableId>,
    pub results: Vec<VariableId>,
}

#[derive(Debug, Clone)]
pub enum Transfer {
    Goto(CfgEdge),
    If {
        condition: VariableId,
        then_edge: CfgEdge,
        else_edge: CfgEdge,
    },
    Return(Vec<VariableId>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgEdge {
    pub target: CfgNodeId,
    pub args: Vec<VariableId>,
}

#[derive(Debug, Clone)]
struct CfgNodeDraft {
    id: CfgNodeId,
    params: Vec<VariableId>,
    block: Vec<BlockInstruction>,
}

#[derive(Debug, Clone)]
pub struct CfgWiring {
    pub node_boundaries: Vec<CfgNodeBoundaries>,
}

#[derive(Debug, Clone)]
pub struct CfgNodeBoundaries {
    pub node: CfgNodeId,
    pub entries: Vec<BoundaryPoint>,
    pub exits: Vec<BoundaryPoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoundaryPoint {
    pub wire: VariableId,
    pub name: Option<VariableName>,
    pub kind: BoundaryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryKind {
    RegionEntry,
    RegionExit,
    FromControl(OperationId),
    ToControl(OperationId),
}

impl Cfg {
    pub fn from_compile_graph(compile_graph: &CompileGraph) -> Result<Self, CfgError> {
        CfgBuilder::new(compile_graph).build()
    }

    pub(super) fn label(&self, node: CfgNodeId) -> String {
        format!("n{node}")
    }
}

#[derive(Debug)]
struct CfgBuilder<'a> {
    compile_graph: &'a CompileGraph,
    node_ids: CfgNodeIdAllocator,
    operation_instances: Vec<OperationInstance>,
    control_operation_ids: Vec<OperationId>,
    data_operation_ids: Vec<OperationId>,
}

impl<'a> CfgBuilder<'a> {
    fn new(compile_graph: &'a CompileGraph) -> Self {
        Self {
            compile_graph,
            node_ids: CfgNodeIdAllocator::default(),
            operation_instances: Vec::new(),
            control_operation_ids: Vec::new(),
            data_operation_ids: Vec::new(),
        }
    }

    fn build(mut self) -> Result<Cfg, CfgError> {
        self.reject_non_data_region()?;
        self.collect_operations();
        Ok(self.build_data_cfg())
    }

    fn reject_non_data_region(&self) -> Result<(), CfgError> {
        match &self.compile_graph.theory {
            CompileTheory::Data => Ok(()),
            other => Err(CfgError::UnsupportedTheory(other.clone())),
        }
    }

    fn collect_operations(&mut self) {
        self.operation_instances = (0..operation_names(self.compile_graph).len())
            .map(|operation_id| operation_instance(self.compile_graph, operation_id))
            .collect();

        for operation in &self.operation_instances {
            if is_control_operation(self.compile_graph, &operation.name) {
                self.control_operation_ids.push(operation.id);
            } else {
                self.data_operation_ids.push(operation.id);
            }
        }
    }

    fn build_data_cfg(&mut self) -> Cfg {
        let boundary = BoundaryWires::from_region_and_control_operations(
            self.compile_graph,
            &self.control_operation_ids,
        );

        let control_fragment = self.control_cfg_fragment();
        let data_fragment = self.data_cfg_fragment(&boundary);

        self.compose_fragments(data_fragment, control_fragment)
    }

    fn control_cfg_fragment(&mut self) -> ControlCfgFragment {
        let expanded_control = ControlExpander::new(self.compile_graph, &self.operation_instances)
            .expand(&self.control_operation_ids);

        let mut node_by_control_operation = HashMap::new();
        let mut control_operation_by_node = HashMap::new();
        let mut node_by_entry_wire = HashMap::new();
        let nodes = expanded_control
            .operations
            .iter()
            .map(|operation| {
                let id = self.node_ids.allocate();
                node_by_control_operation.insert(operation.id, id);
                control_operation_by_node.insert(id, operation.clone());
                for input in &operation.inputs {
                    node_by_entry_wire.insert(*input, id);
                }
                CfgNodeDraft {
                    id,
                    params: operation.inputs.clone(),
                    block: vec![block_instruction(operation.clone())],
                }
            })
            .collect();

        for (visible_operation, entry_operation) in expanded_control.visible_operation_to_entry {
            if let Some(entry_node) = node_by_control_operation.get(&entry_operation).copied() {
                node_by_control_operation.insert(visible_operation, entry_node);
            }
        }

        ControlCfgFragment {
            nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire,
        }
    }

    fn data_cfg_fragment(&mut self, boundary: &BoundaryWires) -> DataCfgFragment {
        let operations_by_cfg_node = data_operations_by_cfg_node(
            self.compile_graph,
            &self.operation_instances,
            &self.data_operation_ids,
            &boundary.all,
        );
        let mut node_by_entry_wire = HashMap::new();
        let mut node_boundaries = Vec::new();

        let nodes = operations_by_cfg_node
            .into_iter()
            .map(|operations| {
                let id = self.node_ids.allocate();
                let (node, boundaries) =
                    data_cfg_node_draft(self.compile_graph, id, operations, boundary);

                for point in &boundaries.entries {
                    node_by_entry_wire.insert(point.wire, id);
                }

                node_boundaries.push(boundaries);
                node
            })
            .collect();

        DataCfgFragment {
            nodes,
            wiring: CfgWiring { node_boundaries },
            node_by_entry_wire,
        }
    }

    fn compose_fragments(
        &self,
        data_fragment: DataCfgFragment,
        control_fragment: ControlCfgFragment,
    ) -> Cfg {
        let DataCfgFragment {
            nodes: data_nodes,
            wiring,
            node_by_entry_wire: data_node_by_entry_wire,
        } = data_fragment;
        let ControlCfgFragment {
            nodes: control_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire: control_node_by_entry_wire,
        } = control_fragment;

        let boundaries_by_node = wiring
            .node_boundaries
            .iter()
            .map(|boundaries| (boundaries.node, boundaries))
            .collect::<HashMap<_, _>>();

        let mut nodes = control_nodes
            .into_iter()
            .map(|node| {
                cfg_node_from_control_draft(
                    node,
                    &control_operation_by_node,
                    &control_node_by_entry_wire,
                    &data_node_by_entry_wire,
                )
            })
            .collect::<Vec<_>>();
        nodes.extend(data_nodes.into_iter().map(|node| {
            let boundaries = boundaries_by_node
                .get(&node.id)
                .expect("data node must have boundary wiring");
            CfgNode {
                id: node.id,
                params: node.params,
                block: node.block,
                transfer: data_transfer(boundaries, &node_by_control_operation),
            }
        }));
        let entry = nodes_with_boundary(&wiring, BoundaryKind::RegionEntry)
            .into_iter()
            .next()
            .or_else(|| nodes.first().map(|node| node.id))
            .unwrap_or(0);
        let predecessors = predecessors(&nodes);

        Cfg {
            entry,
            nodes,
            predecessors,
        }
    }
}

#[derive(Debug, Default)]
struct CfgNodeIdAllocator {
    next: CfgNodeId,
}

#[derive(Debug)]
struct OperationIdAllocator {
    next: OperationId,
}

impl OperationIdAllocator {
    fn new(next: OperationId) -> Self {
        Self { next }
    }

    fn allocate(&mut self) -> OperationId {
        let id = self.next;
        self.next += 1;
        id
    }
}

#[derive(Debug)]
struct VariableIdAllocator {
    next: VariableId,
}

impl VariableIdAllocator {
    fn new(next: VariableId) -> Self {
        Self { next }
    }

    fn allocate(&mut self) -> VariableId {
        let id = self.next;
        self.next += 1;
        id
    }
}

impl CfgNodeIdAllocator {
    fn allocate(&mut self) -> CfgNodeId {
        let id = self.next;
        self.next += 1;
        id
    }
}

#[derive(Debug, Clone)]
struct DataCfgFragment {
    nodes: Vec<CfgNodeDraft>,
    wiring: CfgWiring,
    node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
}

#[derive(Debug, Clone)]
struct ControlCfgFragment {
    nodes: Vec<CfgNodeDraft>,
    node_by_control_operation: HashMap<OperationId, CfgNodeId>,
    control_operation_by_node: HashMap<CfgNodeId, OperationInstance>,
    node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
}

#[derive(Debug, Clone)]
struct ExpandedControlGraph {
    operations: Vec<OperationInstance>,
    visible_operation_to_entry: HashMap<OperationId, OperationId>,
}

#[derive(Debug)]
struct ControlExpander<'a> {
    compile_graph: &'a CompileGraph,
    operation_instances: &'a [OperationInstance],
    operation_ids: OperationIdAllocator,
    variable_ids: VariableIdAllocator,
}

impl<'a> ControlExpander<'a> {
    fn new(compile_graph: &'a CompileGraph, operation_instances: &'a [OperationInstance]) -> Self {
        Self {
            compile_graph,
            operation_instances,
            operation_ids: OperationIdAllocator::new(operation_instances.len()),
            variable_ids: VariableIdAllocator::new(next_variable_id(operation_instances)),
        }
    }

    fn expand(mut self, control_operation_ids: &[OperationId]) -> ExpandedControlGraph {
        let mut operations = Vec::new();
        let mut visible_operation_to_entry = HashMap::new();

        for operation_id in control_operation_ids {
            let call = &self.operation_instances[*operation_id];
            let first = operations.len();
            self.inline_operation(self.compile_graph, call, true, &mut operations);
            if let Some(entry) = operations.get(first) {
                visible_operation_to_entry.insert(call.id, entry.id);
            }
        }

        ExpandedControlGraph {
            operations,
            visible_operation_to_entry,
        }
    }

    fn inline_operation(
        &mut self,
        compile_graph: &CompileGraph,
        call: &OperationInstance,
        keep_call_id: bool,
        output: &mut Vec<OperationInstance>,
    ) {
        let Some(child) = self.child_control_graph(compile_graph, &call.name) else {
            let mut operation = call.clone();
            if !keep_call_id {
                operation.id = self.operation_ids.allocate();
            }
            output.push(operation);
            return;
        };

        let wire_map = self.child_wire_map(child, call);
        for child_operation_id in 0..operation_names(child).len() {
            let child_call = self.remapped_child_operation(child, child_operation_id, &wire_map);
            self.inline_operation(child, &child_call, false, output);
        }
    }

    fn child_control_graph<'b>(
        &self,
        compile_graph: &'b CompileGraph,
        operation: &str,
    ) -> Option<&'b CompileGraph> {
        compile_graph
            .children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.graph)
            .filter(|child| matches!(child.theory, CompileTheory::Control))
    }

    fn child_wire_map(
        &mut self,
        child: &CompileGraph,
        call: &OperationInstance,
    ) -> HashMap<NodeId, VariableId> {
        let mut map = HashMap::new();
        for (wire, variable) in source_nodes(child)
            .into_iter()
            .zip(call.inputs.iter().copied())
        {
            map.insert(wire, variable);
        }
        for (wire, variable) in target_nodes(child)
            .into_iter()
            .zip(call.outputs.iter().copied())
        {
            map.insert(wire, variable);
        }

        for operation in 0..operation_names(child).len() {
            for wire in all_operation_wires(child, operation) {
                map.entry(wire)
                    .or_insert_with(|| self.variable_ids.allocate());
            }
        }

        map
    }

    fn remapped_child_operation(
        &mut self,
        child: &CompileGraph,
        operation_id: OperationId,
        wire_map: &HashMap<NodeId, VariableId>,
    ) -> OperationInstance {
        OperationInstance {
            id: self.operation_ids.allocate(),
            name: operation_names(child)[operation_id].to_string(),
            inputs: operation_sources(child, operation_id)
                .into_iter()
                .filter_map(|wire| wire_map.get(&wire).copied())
                .collect(),
            outputs: operation_targets(child, operation_id)
                .into_iter()
                .filter_map(|wire| wire_map.get(&wire).copied())
                .collect(),
        }
    }
}

fn cfg_node_from_control_draft(
    node: CfgNodeDraft,
    control_operation_by_node: &HashMap<CfgNodeId, OperationInstance>,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> CfgNode {
    let operation = control_operation_by_node
        .get(&node.id)
        .expect("control node must have source operation");
    let transfer = control_transfer(
        node.id,
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
    );
    CfgNode {
        id: node.id,
        params: node.params,
        block: node.block,
        transfer,
    }
}

fn data_cfg_node_draft(
    compile_graph: &CompileGraph,
    id: CfgNodeId,
    operations: Vec<OperationInstance>,
    boundary: &BoundaryWires,
) -> (CfgNodeDraft, CfgNodeBoundaries) {
    let entries = entries_for_node(compile_graph, &operations, boundary);
    let exits = exits_for_node(compile_graph, &operations, boundary);
    let params = entries.iter().map(|entry| entry.wire).collect();
    let block = operations
        .into_iter()
        .map(block_instruction)
        .collect::<Vec<_>>();

    (
        CfgNodeDraft { id, params, block },
        CfgNodeBoundaries {
            node: id,
            entries,
            exits,
        },
    )
}

fn block_instruction(operation: OperationInstance) -> BlockInstruction {
    BlockInstruction {
        operation_id: operation.id,
        operation: operation.name,
        args: operation.inputs,
        results: operation.outputs,
    }
}

fn data_transfer(
    boundaries: &CfgNodeBoundaries,
    control_node_by_operation: &HashMap<OperationId, CfgNodeId>,
) -> Transfer {
    let returns = boundaries
        .exits
        .iter()
        .filter_map(|point| match point.kind {
            BoundaryKind::RegionExit => Some(point.wire),
            BoundaryKind::RegionEntry
            | BoundaryKind::FromControl(_)
            | BoundaryKind::ToControl(_) => None,
        })
        .collect::<Vec<_>>();
    if !returns.is_empty() {
        return Transfer::Return(returns);
    }

    for exit in &boundaries.exits {
        let BoundaryKind::ToControl(control) = exit.kind else {
            continue;
        };
        if let Some(target) = control_node_by_operation.get(&control).copied() {
            let args = boundaries
                .exits
                .iter()
                .filter_map(|point| match point.kind {
                    BoundaryKind::ToControl(point_control) if point_control == control => {
                        Some(point.wire)
                    }
                    BoundaryKind::RegionEntry
                    | BoundaryKind::RegionExit
                    | BoundaryKind::FromControl(_)
                    | BoundaryKind::ToControl(_) => None,
                })
                .collect();
            return Transfer::Goto(CfgEdge { target, args });
        }
    }

    Transfer::Return(Vec::new())
}

fn control_transfer(
    node: CfgNodeId,
    operation: &OperationInstance,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> Transfer {
    let successors = control_successors(
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
    );
    if is_branch_operation(operation) && successors.len() >= 2 {
        return Transfer::If {
            condition: operation.inputs.first().copied().unwrap_or(0),
            then_edge: successors[0].clone(),
            else_edge: successors[1].clone(),
        };
    }
    if is_branch_operation(operation) && operation.outputs.len() >= 2 {
        return Transfer::If {
            condition: operation.inputs.first().copied().unwrap_or(0),
            then_edge: CfgEdge {
                target: node + 1,
                args: vec![operation.outputs[0]],
            },
            else_edge: CfgEdge {
                target: node + 2,
                args: vec![operation.outputs[1]],
            },
        };
    }

    if let Some(edge) = successors.first() {
        return Transfer::Goto(edge.clone());
    }

    Transfer::Return(operation.outputs.clone())
}

fn control_successors(
    operation: &OperationInstance,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> Vec<CfgEdge> {
    let mut successors = Vec::new();
    for output in &operation.outputs {
        if let Some(target) = control_node_by_entry_wire.get(output).copied() {
            push_unique_edge(
                &mut successors,
                CfgEdge {
                    target,
                    args: vec![*output],
                },
            );
        }
        if let Some(target) = data_node_by_entry_wire.get(output).copied() {
            push_unique_edge(
                &mut successors,
                CfgEdge {
                    target,
                    args: vec![*output],
                },
            );
        }
    }
    successors
}

fn is_branch_operation(operation: &OperationInstance) -> bool {
    operation.outputs.len() > 1 || operation.name.contains("branch") || operation.name == "if"
}

fn data_operations_by_cfg_node(
    compile_graph: &CompileGraph,
    operation_instances: &[OperationInstance],
    data_operation_ids: &[OperationId],
    boundary: &HashSet<NodeId>,
) -> Vec<Vec<OperationInstance>> {
    let mut uf = UnionFind::new(operation_names(compile_graph).len());
    let mut internal_wire_to_data_operations = HashMap::<NodeId, Vec<OperationId>>::new();

    for operation_id in data_operation_ids {
        for wire in all_operation_wires(compile_graph, *operation_id) {
            if !boundary.contains(&wire) {
                internal_wire_to_data_operations
                    .entry(wire)
                    .or_default()
                    .push(*operation_id);
            }
        }
    }

    for operations in internal_wire_to_data_operations.values() {
        if let Some((first, rest)) = operations.split_first() {
            for operation in rest {
                uf.union(*first, *operation);
            }
        }
    }

    let mut root_to_cfg_node = HashMap::new();
    let mut operations_by_cfg_node = Vec::<Vec<OperationInstance>>::new();

    for operation_id in data_operation_ids {
        let root = uf.find(*operation_id);
        let next_node = root_to_cfg_node.len();
        let node = *root_to_cfg_node.entry(root).or_insert_with(|| {
            operations_by_cfg_node.push(Vec::new());
            next_node
        });
        operations_by_cfg_node[node].push(operation_instances[*operation_id].clone());
    }

    operations_by_cfg_node
}

#[derive(Debug, Clone)]
struct BoundaryWires {
    all: HashSet<NodeId>,
    control_sources_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
    control_targets_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
}

impl BoundaryWires {
    fn from_region_and_control_operations(
        compile_graph: &CompileGraph,
        control_operation_ids: &[OperationId],
    ) -> Self {
        let mut all = source_nodes(compile_graph)
            .into_iter()
            .collect::<HashSet<_>>();
        all.extend(target_nodes(compile_graph));

        let mut control_sources_by_boundary_wire = HashMap::new();
        let mut control_targets_by_boundary_wire = HashMap::new();

        for operation_id in control_operation_ids {
            for wire in operation_targets(compile_graph, *operation_id) {
                all.insert(wire);
                control_sources_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation_id);
            }

            for wire in operation_sources(compile_graph, *operation_id) {
                all.insert(wire);
                control_targets_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation_id);
            }
        }

        Self {
            all,
            control_sources_by_boundary_wire,
            control_targets_by_boundary_wire,
        }
    }
}

fn entries_for_node(
    compile_graph: &CompileGraph,
    operations: &[OperationInstance],
    boundary: &BoundaryWires,
) -> Vec<BoundaryPoint> {
    let mut entries = Vec::new();
    for operation in operations {
        for wire in operation_sources(compile_graph, operation.id) {
            if !boundary.all.contains(&wire) {
                continue;
            }
            if source_nodes(compile_graph).contains(&wire) {
                push_unique_boundary(
                    &mut entries,
                    boundary_point(compile_graph, wire, BoundaryKind::RegionEntry),
                );
            }
            for control in boundary
                .control_sources_by_boundary_wire
                .get(&wire)
                .into_iter()
                .flatten()
            {
                push_unique_boundary(
                    &mut entries,
                    boundary_point(compile_graph, wire, BoundaryKind::FromControl(*control)),
                );
            }
        }
    }
    entries
}

fn exits_for_node(
    compile_graph: &CompileGraph,
    operations: &[OperationInstance],
    boundary: &BoundaryWires,
) -> Vec<BoundaryPoint> {
    let mut exits = Vec::new();
    for operation in operations {
        for wire in operation_targets(compile_graph, operation.id) {
            if !boundary.all.contains(&wire) {
                continue;
            }
            if target_nodes(compile_graph).contains(&wire) {
                push_unique_boundary(
                    &mut exits,
                    boundary_point(compile_graph, wire, BoundaryKind::RegionExit),
                );
            }
            for control in boundary
                .control_targets_by_boundary_wire
                .get(&wire)
                .into_iter()
                .flatten()
            {
                push_unique_boundary(
                    &mut exits,
                    boundary_point(compile_graph, wire, BoundaryKind::ToControl(*control)),
                );
            }
        }
    }
    exits
}

fn operation_instance(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
) -> OperationInstance {
    OperationInstance {
        id: operation_id,
        name: operation_names(compile_graph)[operation_id].to_string(),
        inputs: variables(&operation_sources(compile_graph, operation_id)),
        outputs: variables(&operation_targets(compile_graph, operation_id)),
    }
}

fn next_variable_id(operation_instances: &[OperationInstance]) -> VariableId {
    operation_instances
        .iter()
        .flat_map(|operation| operation.inputs.iter().chain(&operation.outputs))
        .copied()
        .max()
        .map(|variable| variable + 1)
        .unwrap_or(0)
}

fn is_control_operation(compile_graph: &CompileGraph, operation: &str) -> bool {
    operation.starts_with("control.")
        || compile_graph
            .children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.graph)
            .is_some_and(|child| matches!(child.theory, CompileTheory::Control))
}

fn source_nodes(compile_graph: &CompileGraph) -> Vec<NodeId> {
    compile_graph
        .graph
        .s
        .table
        .iter()
        .copied()
        .map(NodeId)
        .collect()
}

fn target_nodes(compile_graph: &CompileGraph) -> Vec<NodeId> {
    compile_graph
        .graph
        .t
        .table
        .iter()
        .copied()
        .map(NodeId)
        .collect()
}

fn operation_names(compile_graph: &CompileGraph) -> &[crate::lang::Arr] {
    &compile_graph.graph.h.x.0
}

fn operation_sources(compile_graph: &CompileGraph, operation_id: OperationId) -> Vec<NodeId> {
    compile_graph
        .graph
        .h
        .s
        .clone()
        .into_iter()
        .nth(operation_id)
        .map(|sources| sources.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
}

fn operation_targets(compile_graph: &CompileGraph, operation_id: OperationId) -> Vec<NodeId> {
    compile_graph
        .graph
        .h
        .t
        .clone()
        .into_iter()
        .nth(operation_id)
        .map(|targets| targets.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
}

fn variables(nodes: &[NodeId]) -> Vec<VariableId> {
    nodes.iter().map(|node| node.0).collect()
}

fn all_operation_wires(compile_graph: &CompileGraph, operation_id: OperationId) -> Vec<NodeId> {
    let mut wires = operation_sources(compile_graph, operation_id);
    wires.extend(operation_targets(compile_graph, operation_id));
    wires
}

fn boundary_point(compile_graph: &CompileGraph, wire: NodeId, kind: BoundaryKind) -> BoundaryPoint {
    BoundaryPoint {
        wire: wire.0,
        name: compile_graph.source_variable_names.get(&wire.0).cloned(),
        kind,
    }
}

fn push_unique_boundary(target: &mut Vec<BoundaryPoint>, point: BoundaryPoint) {
    if !target.iter().any(|existing| existing == &point) {
        target.push(point);
    }
}

fn push_unique_edge(target: &mut Vec<CfgEdge>, edge: CfgEdge) {
    if !target
        .iter()
        .any(|existing| existing.target == edge.target && existing.args == edge.args)
    {
        target.push(edge);
    }
}

fn nodes_with_boundary(wiring: &CfgWiring, kind: BoundaryKind) -> Vec<CfgNodeId> {
    wiring
        .node_boundaries
        .iter()
        .filter(|boundaries| boundaries.entries.iter().any(|point| point.kind == kind))
        .map(|boundaries| boundaries.node)
        .collect()
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
            Transfer::Return(_) => Vec::new(),
        }
    }
}

fn predecessors(nodes: &[CfgNode]) -> Vec<Vec<CfgNodeId>> {
    let mut predecessors = vec![Vec::new(); nodes.len()];
    for node in nodes {
        for successor in node.successors() {
            predecessors[successor].push(node.id);
        }
    }
    predecessors
}

struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parents: (0..size).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parents[value];
        if parent == value {
            value
        } else {
            let root = self.find(parent);
            self.parents[value] = root;
            root
        }
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.find(left);
        let right = self.find(right);
        if left != right {
            self.parents[right] = left;
        }
    }
}

pub(crate) fn variable_name(id: VariableId) -> String {
    if id > usize::MAX / 2 {
        format!("s{}", usize::MAX - id)
    } else {
        format!("w{id}")
    }
}
