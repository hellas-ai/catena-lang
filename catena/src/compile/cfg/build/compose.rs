use std::collections::HashMap;

use super::normalize::normalize_cfg;
use super::operation::{OperationInstance, is_branch_operation};
use crate::compile::cfg::model::{
    BoundaryKind, Cfg, CfgEdge, CfgNode, CfgNodeBoundaries, CfgNodeDraft, CfgNodeId, CfgWiring,
    OperationId, Transfer, VariableId,
};

pub(super) type BranchTargets = HashMap<OperationId, Vec<CfgEdge>>;

#[derive(Debug, Clone)]
pub(super) struct DataPlan {
    pub(super) nodes: Vec<CfgNodeDraft>,
    pub(super) wiring: CfgWiring,
    pub(super) node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
}

#[derive(Debug, Clone)]
pub(super) struct ControlPlan {
    pub(super) nodes: Vec<CfgNodeDraft>,
    pub(super) nested_data_nodes: Vec<CfgNode>,
    pub(super) node_by_control_operation: HashMap<OperationId, CfgNodeId>,
    pub(super) control_operation_by_node: HashMap<CfgNodeId, OperationInstance>,
    pub(super) node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    pub(super) nested_data_node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    pub(super) branch_targets: BranchTargets,
}

pub(super) fn compose_data_region(data_plan: DataPlan, control_plan: ControlPlan) -> Cfg {
    let DataPlan {
        nodes: data_nodes,
        wiring,
        node_by_entry_wire: data_node_by_entry_wire,
    } = data_plan;
    let ControlPlan {
        nodes: control_nodes,
        nested_data_nodes,
        node_by_control_operation,
        control_operation_by_node,
        node_by_entry_wire: control_node_by_entry_wire,
        nested_data_node_by_entry_wire,
        branch_targets,
    } = control_plan;
    let mut data_node_by_entry_wire = data_node_by_entry_wire;
    data_node_by_entry_wire.extend(nested_data_node_by_entry_wire);
    data_node_by_entry_wire.extend(nested_data_prelude_entries(
        &nested_data_nodes,
        &control_node_by_entry_wire,
    ));

    let mut synthetic_return_nodes = Vec::new();
    let mut next_synthetic_node = control_nodes
        .iter()
        .map(|node| node.id)
        .chain(nested_data_nodes.iter().map(|node| node.id))
        .chain(data_nodes.iter().map(|node| node.id))
        .max()
        .map(|id| id + 1)
        .unwrap_or(0);
    for operation in control_operation_by_node.values() {
        if !is_branch_operation(operation) {
            continue;
        }
        for (index, output) in operation.outputs.iter().copied().enumerate() {
            let has_branch_arm_entrypoint = branch_targets
                .get(&operation.id)
                .is_some_and(|successors| successors.get(index).is_some());
            if has_branch_arm_entrypoint
                || control_node_by_entry_wire.contains_key(&output)
                || data_node_by_entry_wire.contains_key(&output)
            {
                continue;
            }
            let id = next_synthetic_node;
            next_synthetic_node += 1;
            data_node_by_entry_wire.insert(output, id);
            synthetic_return_nodes.push(CfgNode {
                id,
                params: vec![output],
                block: Vec::new(),
                transfer: Transfer::Return(vec![output]),
            });
        }
    }

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
                &branch_targets,
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
    nodes.extend(synthetic_return_nodes);
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

    let entry = data_region_entry(&nodes, &wiring);
    normalize_cfg(Cfg {
        entry,
        nodes,
        predecessors: Vec::new(),
    })
}

pub(super) fn remap_transfer_targets(
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

pub(super) fn predecessors(nodes: &[CfgNode]) -> Vec<Vec<CfgNodeId>> {
    let mut predecessors = vec![Vec::new(); nodes.len()];
    for node in nodes {
        for successor in node.successors() {
            predecessors[successor].push(node.id);
        }
    }
    predecessors
}

fn cfg_node_from_control_draft(
    node: CfgNodeDraft,
    control_operation_by_node: &HashMap<CfgNodeId, OperationInstance>,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    branch_targets: &BranchTargets,
) -> CfgNode {
    let operation = control_operation_by_node
        .get(&node.id)
        .expect("control node must have source operation");
    let transfer = control_transfer(
        node.id,
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
        branch_targets,
    );
    CfgNode {
        id: node.id,
        params: node.params,
        block: node.block,
        transfer,
    }
}

fn nested_data_prelude_entries(
    nested_data_nodes: &[CfgNode],
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> HashMap<VariableId, CfgNodeId> {
    let mut entries = HashMap::new();
    for node in nested_data_nodes {
        let target = match &node.transfer {
            Transfer::Goto(edge) => Some(edge.target),
            Transfer::Return(values) => values
                .iter()
                .find_map(|value| control_node_by_entry_wire.get(value).copied()),
            Transfer::If { .. } => None,
        };
        let Some(target) = target else {
            continue;
        };
        for (wire, control_node) in control_node_by_entry_wire {
            if *control_node == target {
                entries.insert(*wire, node.id);
            }
        }
    }
    entries
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
    branch_targets: &BranchTargets,
) -> Transfer {
    if let Some(successors) = branch_targets.get(&operation.id)
        && successors.len() >= 2
    {
        return Transfer::If {
            condition: branch_condition(operation),
            then_edge: successors[0].clone(),
            else_edge: successors[1].clone(),
        };
    }

    let successors = control_successors(
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
    );
    if is_branch_operation(operation) && successors.len() >= 2 {
        return Transfer::If {
            condition: branch_condition(operation),
            then_edge: successors[0].clone(),
            else_edge: successors[1].clone(),
        };
    }
    if is_branch_operation(operation) && operation.outputs.len() >= 2 {
        return Transfer::If {
            condition: branch_condition(operation),
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

fn data_region_entry(nodes: &[CfgNode], wiring: &CfgWiring) -> CfgNodeId {
    let region_entries = nodes_with_boundary(wiring, BoundaryKind::RegionEntry);
    region_entries
        .iter()
        .copied()
        .find(|entry| node_is_non_empty(nodes, *entry))
        .or_else(|| region_entries.into_iter().next())
        .or_else(|| first_node(nodes))
        .unwrap_or(0)
}

fn nodes_with_boundary(wiring: &CfgWiring, kind: BoundaryKind) -> Vec<CfgNodeId> {
    wiring
        .node_boundaries
        .iter()
        .filter(|boundaries| boundaries.entries.iter().any(|point| point.kind == kind))
        .map(|boundaries| boundaries.node)
        .collect()
}

fn node_is_non_empty(nodes: &[CfgNode], id: CfgNodeId) -> bool {
    nodes
        .iter()
        .find(|node| node.id == id)
        .is_some_and(|node| !node.block.is_empty())
}

fn first_node(nodes: &[CfgNode]) -> Option<CfgNodeId> {
    nodes.first().map(|node| node.id)
}

fn remap_edge_target(edge: CfgEdge, node_id_by_old: &HashMap<CfgNodeId, CfgNodeId>) -> CfgEdge {
    CfgEdge {
        target: node_id_by_old[&edge.target],
        args: edge.args,
    }
}

fn branch_condition(operation: &OperationInstance) -> VariableId {
    operation
        .branch_condition
        .or_else(|| operation.inputs.first().copied())
        .unwrap_or(0)
}

fn push_unique_edge(target: &mut Vec<CfgEdge>, edge: CfgEdge) {
    if !target
        .iter()
        .any(|existing| existing.target == edge.target && existing.args == edge.args)
    {
        target.push(edge);
    }
}
