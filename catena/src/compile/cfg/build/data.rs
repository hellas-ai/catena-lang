use std::collections::{HashMap, HashSet};

use open_hypergraphs::lax::NodeId;

use crate::compile::CompileGraph;

use super::{
    monoidal::MonoidalStructureResolver,
    operation::{
        CfgOperationRole, OperationInstance, cfg_operation_role, local_operation_name, mapped_wire,
        operation_names, source_nodes, target_nodes,
    },
};
use crate::compile::cfg::model::{
    BlockInstruction, BoundaryKind, BoundaryPoint, CfgError, CfgNodeBoundaries, CfgNodeDraft,
    CfgNodeId, OperationId, VariableId,
};

#[derive(Debug, Clone)]
pub(super) struct DataBoundaries {
    pub(super) all: HashSet<NodeId>,
    region_sources: HashSet<NodeId>,
    region_targets: HashSet<NodeId>,
    control_sources_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
    control_targets_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
}

impl DataBoundaries {
    pub(super) fn from_region_and_control_operations(
        compile_graph: &CompileGraph,
        control_operations: &[OperationInstance],
        wire_map: &HashMap<NodeId, VariableId>,
        monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    ) -> Self {
        let region_sources = region_boundary_wires(
            source_nodes(compile_graph),
            wire_map,
            monoidal_structure_resolver,
        );
        let region_targets = target_nodes(compile_graph)
            .into_iter()
            .map(|wire| NodeId(mapped_wire(wire, wire_map)))
            .collect::<HashSet<_>>();
        let mut all = region_sources.clone();
        all.extend(region_targets.iter().copied());

        let mut control_sources_by_boundary_wire = HashMap::new();
        let mut control_targets_by_boundary_wire = HashMap::new();

        for operation in control_operations {
            for wire in &operation.outputs {
                let wire = NodeId(*wire);
                all.insert(wire);
                control_sources_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(operation.id);
            }

            for wire in control_input_boundary_wires(operation) {
                let wire = NodeId(wire);
                all.insert(wire);
                control_targets_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(operation.id);
            }
        }

        Self {
            all,
            region_sources,
            region_targets,
            control_sources_by_boundary_wire,
            control_targets_by_boundary_wire,
        }
    }
}

fn region_boundary_wires(
    wires: Vec<NodeId>,
    wire_map: &HashMap<NodeId, VariableId>,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
) -> HashSet<NodeId> {
    let mut result = HashSet::new();
    for wire in wires {
        let wire = mapped_wire(wire, wire_map);
        result.insert(NodeId(wire));
        result.extend(
            monoidal_structure_resolver
                .atom_variables(wire)
                .into_iter()
                .map(NodeId),
        );
    }
    result
}

fn control_input_boundary_wires(operation: &OperationInstance) -> Vec<VariableId> {
    let mut wires = operation.inputs.clone();
    if let Some(condition) = operation.branch_condition
        && !wires.contains(&condition)
    {
        wires.push(condition);
    }
    wires
}

fn entries_for_node(
    compile_graph: &CompileGraph,
    operations: &[OperationInstance],
    boundary: &DataBoundaries,
) -> Vec<BoundaryPoint> {
    let mut entries = Vec::new();
    for operation in operations {
        for wire in &operation.inputs {
            let wire = NodeId(*wire);
            if !boundary.all.contains(&wire) {
                continue;
            }
            if boundary.region_sources.contains(&wire) {
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
    boundary: &DataBoundaries,
) -> Vec<BoundaryPoint> {
    let mut exits = Vec::new();
    for operation in operations {
        for wire in &operation.outputs {
            let wire = NodeId(*wire);
            if !boundary.all.contains(&wire) {
                continue;
            }
            if boundary.region_targets.contains(&wire) {
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

pub(super) fn data_cfg_node_draft(
    compile_graph: &CompileGraph,
    id: CfgNodeId,
    operations: Vec<OperationInstance>,
    boundary: &DataBoundaries,
) -> Result<(CfgNodeDraft, CfgNodeBoundaries), CfgError> {
    let entries = entries_for_node(compile_graph, &operations, boundary);
    let exits = exits_for_node(compile_graph, &operations, boundary);
    let block = operations
        .into_iter()
        .map(block_instruction)
        .filter_map(Result::transpose)
        .collect::<Result<Vec<_>, CfgError>>()?;
    let used_inputs = block
        .iter()
        .flat_map(|instruction| instruction.args.iter().copied())
        .collect::<HashSet<_>>();
    let params = entries
        .iter()
        .filter_map(|entry| used_inputs.contains(&entry.wire).then_some(entry.wire))
        .collect();

    Ok((
        CfgNodeDraft { id, params, block },
        CfgNodeBoundaries {
            node: id,
            entries,
            exits,
        },
    ))
}

pub(super) fn block_instructions(
    operation: OperationInstance,
) -> Result<Vec<BlockInstruction>, CfgError> {
    Ok(block_instruction(operation)?.into_iter().collect())
}

pub(super) fn control_region_block_instructions(
    operation: OperationInstance,
) -> Result<Vec<BlockInstruction>, CfgError> {
    if local_operation_name(&operation.name) == "never" {
        return Ok(vec![BlockInstruction {
            operation_id: operation.id,
            operation: operation.name,
            args: operation.inputs,
            results: Vec::new(),
        }]);
    }
    block_instructions(operation)
}

pub(super) fn block_instruction(
    operation: OperationInstance,
) -> Result<Option<BlockInstruction>, CfgError> {
    match cfg_operation_role(&operation.name) {
        CfgOperationRole::Instruction => Ok(Some(BlockInstruction {
            operation_id: operation.id,
            operation: operation.name,
            args: operation.inputs,
            results: operation.outputs,
        })),
        CfgOperationRole::MonoidalStructure | CfgOperationRole::ControlFlow => Ok(None),
    }
}
// Data operation partitioning

pub(super) fn partition_data_operations_by_internal_wires(
    compile_graph: &CompileGraph,
    operation_instances: &[OperationInstance],
    data_operation_ids: &[OperationId],
    boundary: &HashSet<NodeId>,
) -> Vec<Vec<OperationInstance>> {
    let mut uf = UnionFind::new(operation_names(compile_graph).len());
    let mut internal_wire_to_data_operations = HashMap::<NodeId, Vec<OperationId>>::new();

    for operation_id in data_operation_ids {
        for wire in operation_instances[*operation_id]
            .inputs
            .iter()
            .chain(&operation_instances[*operation_id].outputs)
            .copied()
            .map(NodeId)
        {
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
