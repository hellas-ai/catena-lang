use crate::compile::{CompileGraph, CompileTheory};
use open_hypergraphs::lax::NodeId;
use std::collections::{HashMap, HashSet};

// CFG model

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

// CFG construction

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
        self.build_data_cfg()
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

    fn build_data_cfg(&mut self) -> Result<Cfg, CfgError> {
        let boundary = BoundaryWires::from_region_and_control_operations(
            self.compile_graph,
            &self.control_operation_ids,
        );

        let control_fragment = self.control_cfg_fragment()?;
        let data_fragment = self.data_cfg_fragment(&boundary);

        Ok(self.compose_fragments(data_fragment, control_fragment))
    }

    fn control_cfg_fragment(&mut self) -> Result<ControlCfgFragment, CfgError> {
        let expanded_control = ControlExpander::new(self.compile_graph, &self.operation_instances)
            .expand(&self.control_operation_ids)?;

        let mut node_by_control_operation = HashMap::new();
        let mut control_operation_by_node = HashMap::new();
        let mut node_by_entry_wire = HashMap::new();
        let mut nested_data_nodes = Vec::new();
        let mut nested_data_node_by_entry_wire = HashMap::new();
        let mut branch_data_successors = HashMap::<OperationId, Vec<CfgEdge>>::new();
        let mut current_branch = None::<OperationInstance>;
        let mut nodes = Vec::new();

        for item in expanded_control.items {
            match item {
                ExpandedControlItem::Control(operation) => {
                    let id = self.node_ids.allocate();
                    node_by_control_operation.insert(operation.id, id);
                    control_operation_by_node.insert(id, operation.clone());
                    for input in &operation.inputs {
                        node_by_entry_wire.insert(*input, id);
                    }
                    nodes.push(CfgNodeDraft {
                        id,
                        params: operation.inputs.clone(),
                        block: vec![block_instruction(operation)],
                    });
                    current_branch = control_operation_by_node
                        .get(&id)
                        .filter(|operation| is_branch_operation(operation))
                        .cloned();
                }
                ExpandedControlItem::DataCfg { call, cfg } => {
                    let remapped_cfg = self.remap_cfg_nodes(cfg);
                    if let Some(entry) = remapped_cfg
                        .nodes
                        .iter()
                        .find(|node| node.id == remapped_cfg.entry)
                    {
                        for input in &call.inputs {
                            nested_data_node_by_entry_wire.insert(*input, entry.id);
                        }
                        if let Some(branch) = &current_branch {
                            let successors = branch_data_successors.entry(branch.id).or_default();
                            let arg = branch
                                .outputs
                                .get(successors.len())
                                .copied()
                                .or_else(|| call.inputs.first().copied())
                                .into_iter()
                                .collect();
                            successors.push(CfgEdge {
                                target: entry.id,
                                args: arg,
                            });
                        }
                    }
                    nested_data_nodes.extend(remapped_cfg.nodes);
                }
            }
        }

        for (visible_operation, entry_operation) in expanded_control.visible_operation_to_entry {
            if let Some(entry_node) = node_by_control_operation.get(&entry_operation).copied() {
                node_by_control_operation.insert(visible_operation, entry_node);
            }
        }

        Ok(ControlCfgFragment {
            nodes,
            nested_data_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_data_successors,
        })
    }

    fn remap_cfg_nodes(&mut self, mut cfg: Cfg) -> Cfg {
        let node_id_by_old = cfg
            .nodes
            .iter()
            .map(|node| (node.id, self.node_ids.allocate()))
            .collect::<HashMap<_, _>>();

        for node in &mut cfg.nodes {
            node.id = node_id_by_old[&node.id];
            node.transfer = remap_transfer_targets(node.transfer.clone(), &node_id_by_old);
        }
        cfg.entry = node_id_by_old[&cfg.entry];
        cfg
    }

    fn data_cfg_fragment(&mut self, boundary: &BoundaryWires) -> DataCfgFragment {
        let operations_by_cfg_node = partition_data_operations_by_internal_wires(
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
            nested_data_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire: control_node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_data_successors,
        } = control_fragment;
        let mut data_node_by_entry_wire = data_node_by_entry_wire;
        data_node_by_entry_wire.extend(nested_data_node_by_entry_wire);

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
                    &branch_data_successors,
                )
            })
            .collect::<Vec<_>>();
        nodes.extend(nested_data_nodes.into_iter().map(|mut node| {
            node.transfer = resolve_nested_data_return(
                node.transfer,
                &control_node_by_entry_wire,
                &data_node_by_entry_wire,
            );
            node
        }));
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
        nodes.sort_by_key(|node| node.id);
        let predecessors = predecessors(&nodes);

        Cfg {
            entry,
            nodes,
            predecessors,
        }
    }
}

// CFG construction state

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
    nested_data_nodes: Vec<CfgNode>,
    node_by_control_operation: HashMap<OperationId, CfgNodeId>,
    control_operation_by_node: HashMap<CfgNodeId, OperationInstance>,
    node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    nested_data_node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    branch_data_successors: HashMap<OperationId, Vec<CfgEdge>>,
}

// Control expansion

#[derive(Debug, Clone)]
struct ExpandedControlGraph {
    items: Vec<ExpandedControlItem>,
    visible_operation_to_entry: HashMap<OperationId, OperationId>,
}

#[derive(Debug, Clone)]
enum ExpandedControlItem {
    Control(OperationInstance),
    DataCfg { call: OperationInstance, cfg: Cfg },
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

    fn expand(
        mut self,
        control_operation_ids: &[OperationId],
    ) -> Result<ExpandedControlGraph, CfgError> {
        let mut items = Vec::new();
        let mut visible_operation_to_entry = HashMap::new();

        for operation_id in control_operation_ids {
            let call = &self.operation_instances[*operation_id];
            let first = items.len();
            self.inline_operation(self.compile_graph, call, true, &mut items)?;
            if let Some(ExpandedControlItem::Control(entry)) = items.get(first) {
                visible_operation_to_entry.insert(call.id, entry.id);
            }
        }

        Ok(ExpandedControlGraph {
            items,
            visible_operation_to_entry,
        })
    }

    fn inline_operation(
        &mut self,
        compile_graph: &CompileGraph,
        call: &OperationInstance,
        keep_call_id: bool,
        output: &mut Vec<ExpandedControlItem>,
    ) -> Result<(), CfgError> {
        let Some(child) = self.child_control_graph(compile_graph, &call.name) else {
            if let Some(child) = child_data_graph_for_operation(compile_graph, &call.name) {
                let cfg = self.remapped_data_cfg(child, call)?;
                output.push(ExpandedControlItem::DataCfg {
                    call: call.clone(),
                    cfg,
                });
                return Ok(());
            }
            let mut operation = call.clone();
            if !keep_call_id {
                operation.id = self.operation_ids.allocate();
            }
            output.push(ExpandedControlItem::Control(operation));
            return Ok(());
        };

        let wire_map = self.child_wire_map(child, call);
        for child_operation_id in 0..operation_names(child).len() {
            let child_call = self.remapped_child_operation(child, child_operation_id, &wire_map);
            self.inline_operation(child, &child_call, false, output)?;
        }
        Ok(())
    }

    fn child_control_graph(
        &self,
        compile_graph: &'a CompileGraph,
        operation: &str,
    ) -> Option<&'a CompileGraph> {
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

    fn remapped_data_cfg(
        &mut self,
        child: &CompileGraph,
        call: &OperationInstance,
    ) -> Result<Cfg, CfgError> {
        let mut cfg = Cfg::from_compile_graph(child)?;
        let mut variable_map = self.child_wire_map(child, call);
        for variable in cfg_variables(&cfg) {
            variable_map
                .entry(NodeId(variable))
                .or_insert_with(|| self.variable_ids.allocate());
        }
        remap_cfg_variables(&mut cfg, &variable_map);
        Ok(cfg)
    }
}

fn child_data_graph_for_operation<'a>(
    compile_graph: &'a CompileGraph,
    operation: &str,
) -> Option<&'a CompileGraph> {
    let local_name = operation.strip_prefix("data.");
    compile_graph
        .children
        .iter()
        .find(|child| {
            child.operation == operation
                || local_name.is_some_and(|local_name| child.graph.definition_name == local_name)
        })
        .map(|child| &child.graph)
        .filter(|child| matches!(child.theory, CompileTheory::Data))
}

// CFG node drafts

fn cfg_node_from_control_draft(
    node: CfgNodeDraft,
    control_operation_by_node: &HashMap<CfgNodeId, OperationInstance>,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    branch_data_successors: &HashMap<OperationId, Vec<CfgEdge>>,
) -> CfgNode {
    let operation = control_operation_by_node
        .get(&node.id)
        .expect("control node must have source operation");
    let transfer = control_transfer(
        node.id,
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
        branch_data_successors,
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

// Transfers

fn remap_transfer_targets(
    transfer: Transfer,
    node_id_by_old: &HashMap<CfgNodeId, CfgNodeId>,
) -> Transfer {
    match transfer {
        Transfer::Goto(edge) => Transfer::Goto(remap_edge_target(edge, node_id_by_old)),
        Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => Transfer::If {
            condition,
            then_edge: remap_edge_target(then_edge, node_id_by_old),
            else_edge: remap_edge_target(else_edge, node_id_by_old),
        },
        Transfer::Return(values) => Transfer::Return(values),
    }
}

fn remap_edge_target(edge: CfgEdge, node_id_by_old: &HashMap<CfgNodeId, CfgNodeId>) -> CfgEdge {
    CfgEdge {
        target: node_id_by_old[&edge.target],
        args: edge.args,
    }
}

fn resolve_nested_data_return(
    transfer: Transfer,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> Transfer {
    let Transfer::Return(values) = transfer else {
        return transfer;
    };
    let Some(target) = values
        .iter()
        .find_map(|value| control_node_by_entry_wire.get(value).copied())
        .or_else(|| {
            values
                .iter()
                .find_map(|value| data_node_by_entry_wire.get(value).copied())
        })
    else {
        return Transfer::Return(values);
    };
    Transfer::Goto(CfgEdge {
        target,
        args: values,
    })
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
    branch_data_successors: &HashMap<OperationId, Vec<CfgEdge>>,
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
    if let Some(successors) = branch_data_successors.get(&operation.id)
        && successors.len() >= 2
    {
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

// Data operation partitioning

fn partition_data_operations_by_internal_wires(
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

// Boundary wiring

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

// Compile graph accessors

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

fn cfg_variables(cfg: &Cfg) -> Vec<VariableId> {
    let mut variables = Vec::new();
    for node in &cfg.nodes {
        variables.extend(node.params.iter().copied());
        for instruction in &node.block {
            variables.extend(instruction.args.iter().copied());
            variables.extend(instruction.results.iter().copied());
        }
        match &node.transfer {
            Transfer::Goto(edge) => variables.extend(edge.args.iter().copied()),
            Transfer::If {
                condition,
                then_edge,
                else_edge,
            } => {
                variables.push(*condition);
                variables.extend(then_edge.args.iter().copied());
                variables.extend(else_edge.args.iter().copied());
            }
            Transfer::Return(values) => variables.extend(values.iter().copied()),
        }
    }
    variables
}

fn remap_cfg_variables(cfg: &mut Cfg, variable_map: &HashMap<NodeId, VariableId>) {
    for node in &mut cfg.nodes {
        remap_variables(&mut node.params, variable_map);
        for instruction in &mut node.block {
            remap_variables(&mut instruction.args, variable_map);
            remap_variables(&mut instruction.results, variable_map);
        }
        match &mut node.transfer {
            Transfer::Goto(edge) => remap_variables(&mut edge.args, variable_map),
            Transfer::If {
                condition,
                then_edge,
                else_edge,
            } => {
                *condition = variable_map[&NodeId(*condition)];
                remap_variables(&mut then_edge.args, variable_map);
                remap_variables(&mut else_edge.args, variable_map);
            }
            Transfer::Return(values) => remap_variables(values, variable_map),
        }
    }
}

fn remap_variables(variables: &mut [VariableId], variable_map: &HashMap<NodeId, VariableId>) {
    for variable in variables {
        *variable = variable_map[&NodeId(*variable)];
    }
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

// Union-find

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

// Variable naming

pub(crate) fn variable_name(id: VariableId) -> String {
    if id > usize::MAX / 2 {
        format!("s{}", usize::MAX - id)
    } else {
        format!("w{id}")
    }
}
