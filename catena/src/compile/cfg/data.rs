use std::collections::{HashMap, HashSet};

use open_hypergraphs::lax::NodeId;

use crate::compile::CompileGraph;

use super::{
    model::{
        BlockInstruction, CfgError, CfgNodeBoundaries, CfgNodeDraft, CfgNodeId, CfgOptions,
        OperationId,
    },
    operation::{CfgOperationRole, OperationInstance, cfg_operation_role, operation_names},
    wiring::{BoundaryWires, entries_for_node, exits_for_node},
};

pub(super) fn data_cfg_node_draft(
    compile_graph: &CompileGraph,
    id: CfgNodeId,
    operations: Vec<OperationInstance>,
    boundary: &BoundaryWires,
    options: CfgOptions,
) -> Result<(CfgNodeDraft, CfgNodeBoundaries), CfgError> {
    let entries = entries_for_node(compile_graph, &operations, boundary);
    let exits = exits_for_node(compile_graph, &operations, boundary);
    let block = operations
        .into_iter()
        .map(|operation| block_instruction(operation, options))
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
    options: CfgOptions,
) -> Result<Vec<BlockInstruction>, CfgError> {
    Ok(block_instruction(operation, options)?.into_iter().collect())
}

pub(super) fn block_instruction(
    operation: OperationInstance,
    options: CfgOptions,
) -> Result<Option<BlockInstruction>, CfgError> {
    match cfg_operation_role(&operation.name) {
        CfgOperationRole::Instruction => Ok(Some(block_instruction_from_operation(operation))),
        CfgOperationRole::MonoidalStructure if options.keep_monoidal_operations => {
            Ok(Some(block_instruction_from_operation(operation)))
        }
        CfgOperationRole::ControlFlow if options.keep_control_flow_operations => {
            Ok(Some(block_instruction_from_operation(operation)))
        }
        CfgOperationRole::MonoidalStructure | CfgOperationRole::ControlFlow => Ok(None),
    }
}

fn block_instruction_from_operation(operation: OperationInstance) -> BlockInstruction {
    BlockInstruction {
        operation_id: operation.id,
        operation: operation.name,
        args: operation.inputs,
        results: operation.outputs,
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
