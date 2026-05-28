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
pub enum StructuredError {
    #[error("cfg2 only accepts data regions; got {0}")]
    UnsupportedTheory(CompileTheory),
}

#[derive(Debug, Clone)]
pub struct CfgClassification {
    pub nodes: Vec<ClassifiedCfgNode>,
    pub control_transfers: Vec<ClassifiedControlTransfer>,
    pub operation_classes: Vec<OperationClass>,
    pub entry_nodes: Vec<CfgNodeId>,
    pub exit_nodes: Vec<CfgNodeId>,
}

#[derive(Debug, Clone)]
pub struct ClassifiedCfgNode {
    pub id: CfgNodeId,
    pub operations: Vec<OperationInstance>,
    pub entries: Vec<BoundaryPoint>,
    pub exits: Vec<BoundaryPoint>,
    pub incoming_control: Vec<OperationId>,
    pub outgoing_control: Vec<OperationId>,
}

#[derive(Debug, Clone)]
pub struct ClassifiedControlTransfer {
    pub operation: OperationInstance,
    pub source_nodes: Vec<CfgNodeId>,
    pub target_nodes: Vec<CfgNodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationClass {
    DataNode(CfgNodeId),
    ControlTransfer(usize),
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

impl CfgClassification {
    pub fn from_compile_graph(compile_graph: &CompileGraph) -> Result<Self, StructuredError> {
        match &compile_graph.theory {
            CompileTheory::Data => Ok(classify_data_region(compile_graph)),
            other => Err(StructuredError::UnsupportedTheory(other.clone())),
        }
    }
}

fn classify_data_region(compile_graph: &CompileGraph) -> CfgClassification {
    let operation_instances = (0..operation_names(compile_graph).len())
        .map(|operation_id| operation_instance(compile_graph, operation_id))
        .collect::<Vec<_>>();

    let mut control_operations = Vec::new();
    let mut data_operations = Vec::new();
    for operation in &operation_instances {
        if is_control_operation(compile_graph, &operation.name) {
            control_operations.push(operation.id);
        } else {
            data_operations.push(operation.id);
        }
    }

    let boundary =
        BoundaryWires::from_region_and_control_operations(compile_graph, &control_operations);
    let data_node_candidates = data_node_candidates_from_internal_wires(
        compile_graph,
        &operation_instances,
        &data_operations,
        &boundary.all,
    );

    let classified_data_nodes = classify_data_node_candidates(
        compile_graph,
        data_node_candidates.operations_by_cfg_node,
        &boundary,
    );

    let mut operation_classes = vec![OperationClass::ControlTransfer(0); operation_instances.len()];
    for (operation, node) in data_node_candidates.cfg_node_for_operation {
        operation_classes[operation] = OperationClass::DataNode(node);
    }

    let mut control_transfers = Vec::new();
    for control in control_operations {
        let transfer_id = control_transfers.len();
        operation_classes[control] = OperationClass::ControlTransfer(transfer_id);
        control_transfers.push(ClassifiedControlTransfer {
            operation: operation_instances[control].clone(),
            source_nodes: unique_nodes_for_wires(
                &operation_instances[control].inputs,
                &classified_data_nodes.cfg_node_by_exit_wire,
            ),
            target_nodes: unique_nodes_for_wires(
                &operation_instances[control].outputs,
                &classified_data_nodes.cfg_node_by_entry_wire,
            ),
        });
    }

    CfgClassification {
        entry_nodes: nodes_with_boundary(&classified_data_nodes.nodes, BoundaryKind::RegionEntry),
        exit_nodes: nodes_with_boundary(&classified_data_nodes.nodes, BoundaryKind::RegionExit),
        nodes: classified_data_nodes.nodes,
        control_transfers,
        operation_classes,
    }
}

#[derive(Debug, Clone)]
struct ClassifiedDataNodes {
    nodes: Vec<ClassifiedCfgNode>,
    cfg_node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    cfg_node_by_exit_wire: HashMap<VariableId, CfgNodeId>,
}

fn classify_data_node_candidates(
    compile_graph: &CompileGraph,
    operations_by_cfg_node: Vec<Vec<OperationInstance>>,
    boundary: &BoundaryWires,
) -> ClassifiedDataNodes {
    let mut cfg_node_by_entry_wire = HashMap::new();
    let mut cfg_node_by_exit_wire = HashMap::new();

    let nodes = operations_by_cfg_node
        .into_iter()
        .enumerate()
        .map(|(id, operations)| {
            let node = classified_data_node(compile_graph, id, operations, boundary);

            for point in &node.entries {
                cfg_node_by_entry_wire.insert(point.wire, id);
            }
            for point in &node.exits {
                cfg_node_by_exit_wire.insert(point.wire, id);
            }

            node
        })
        .collect();

    ClassifiedDataNodes {
        nodes,
        cfg_node_by_entry_wire,
        cfg_node_by_exit_wire,
    }
}

fn classified_data_node(
    compile_graph: &CompileGraph,
    id: CfgNodeId,
    operations: Vec<OperationInstance>,
    boundary: &BoundaryWires,
) -> ClassifiedCfgNode {
    let entries = entries_for_node(compile_graph, &operations, boundary);
    let exits = exits_for_node(compile_graph, &operations, boundary);

    ClassifiedCfgNode {
        id,
        operations,
        incoming_control: incoming_control_operations(&entries),
        outgoing_control: outgoing_control_operations(&exits),
        entries,
        exits,
    }
}

fn incoming_control_operations(entries: &[BoundaryPoint]) -> Vec<OperationId> {
    sorted_ids(entries.iter().filter_map(|point| match point.kind {
        BoundaryKind::FromControl(control) => Some(control),
        BoundaryKind::RegionEntry | BoundaryKind::RegionExit | BoundaryKind::ToControl(_) => None,
    }))
}

fn outgoing_control_operations(exits: &[BoundaryPoint]) -> Vec<OperationId> {
    sorted_ids(exits.iter().filter_map(|point| match point.kind {
        BoundaryKind::ToControl(control) => Some(control),
        BoundaryKind::RegionEntry | BoundaryKind::RegionExit | BoundaryKind::FromControl(_) => None,
    }))
}

#[derive(Debug, Clone)]
struct DataNodeCandidates {
    cfg_node_for_operation: HashMap<OperationId, CfgNodeId>,
    operations_by_cfg_node: Vec<Vec<OperationInstance>>,
}

fn data_node_candidates_from_internal_wires(
    compile_graph: &CompileGraph,
    operation_instances: &[OperationInstance],
    data_operations: &[OperationId],
    boundary: &HashSet<NodeId>,
) -> DataNodeCandidates {
    let mut uf = UnionFind::new(operation_names(compile_graph).len());
    let mut internal_wire_to_data_operations = HashMap::<NodeId, Vec<OperationId>>::new();

    for operation in data_operations {
        for wire in all_operation_wires(compile_graph, *operation) {
            if !boundary.contains(&wire) {
                internal_wire_to_data_operations
                    .entry(wire)
                    .or_default()
                    .push(*operation);
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
    let mut cfg_node_for_operation = HashMap::new();
    let mut operations_by_cfg_node = Vec::<Vec<OperationInstance>>::new();

    for operation in data_operations {
        let root = uf.find(*operation);
        let next_node = root_to_cfg_node.len();
        let node = *root_to_cfg_node.entry(root).or_insert_with(|| {
            operations_by_cfg_node.push(Vec::new());
            next_node
        });
        cfg_node_for_operation.insert(*operation, node);
        operations_by_cfg_node[node].push(operation_instances[*operation].clone());
    }

    DataNodeCandidates {
        cfg_node_for_operation,
        operations_by_cfg_node,
    }
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
        control_operations: &[OperationId],
    ) -> Self {
        let mut all = source_nodes(compile_graph)
            .into_iter()
            .collect::<HashSet<_>>();
        all.extend(target_nodes(compile_graph));

        let mut control_sources_by_boundary_wire = HashMap::new();
        let mut control_targets_by_boundary_wire = HashMap::new();

        for operation in control_operations {
            for wire in operation_targets(compile_graph, *operation) {
                all.insert(wire);
                control_sources_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation);
            }

            for wire in operation_sources(compile_graph, *operation) {
                all.insert(wire);
                control_targets_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation);
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

fn unique_nodes_for_wires(
    wires: &[VariableId],
    node_by_wire: &HashMap<VariableId, CfgNodeId>,
) -> Vec<CfgNodeId> {
    let mut nodes = wires
        .iter()
        .filter_map(|wire| node_by_wire.get(wire).copied())
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn nodes_with_boundary(nodes: &[ClassifiedCfgNode], kind: BoundaryKind) -> Vec<CfgNodeId> {
    nodes
        .iter()
        .filter(|node| {
            node.entries
                .iter()
                .chain(&node.exits)
                .any(|point| point.kind == kind)
        })
        .map(|node| node.id)
        .collect()
}

fn sorted_ids(ids: impl Iterator<Item = OperationId>) -> Vec<OperationId> {
    let mut ids = ids.collect::<HashSet<_>>().into_iter().collect::<Vec<_>>();
    ids.sort_unstable();
    ids
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
