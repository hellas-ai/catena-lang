use std::collections::{HashMap, HashSet};

use open_hypergraphs::lax::NodeId;

use crate::compile::CompileGraph;

use super::{
    model::{
        BoundaryKind, BoundaryPoint, CfgEdge, CfgNode, CfgNodeBoundaries, CfgNodeDraft, CfgNodeId,
        CfgWiring, OperationId, Transfer, VariableId,
    },
    operation::{OperationInstance, mapped_wire, source_nodes, target_nodes},
};

// CFG node drafts

pub(super) fn cfg_node_from_control_draft(
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
// Transfers

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

fn remap_edge_target(edge: CfgEdge, node_id_by_old: &HashMap<CfgNodeId, CfgNodeId>) -> CfgEdge {
    CfgEdge {
        target: node_id_by_old[&edge.target],
        args: edge.args,
    }
}

pub(super) fn resolve_nested_data_return(
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

pub(super) fn data_transfer(
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

pub(super) fn control_transfer(
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
            condition: branch_condition(operation),
            then_edge: successors[0].clone(),
            else_edge: successors[1].clone(),
        };
    }
    if let Some(successors) = branch_data_successors.get(&operation.id)
        && successors.len() >= 2
    {
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

pub(super) fn control_successors(
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

fn branch_condition(operation: &OperationInstance) -> VariableId {
    operation
        .branch_condition
        .or_else(|| operation.inputs.first().copied())
        .unwrap_or(0)
}
// Boundary wiring

#[derive(Debug, Clone)]
pub(super) struct BoundaryWires {
    pub(super) all: HashSet<NodeId>,
    pub(super) region_sources: HashSet<NodeId>,
    pub(super) region_targets: HashSet<NodeId>,
    pub(super) control_sources_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
    pub(super) control_targets_by_boundary_wire: HashMap<NodeId, Vec<OperationId>>,
}

impl BoundaryWires {
    pub(super) fn from_region_and_control_operations(
        compile_graph: &CompileGraph,
        operation_instances: &[OperationInstance],
        control_operation_ids: &[OperationId],
        wire_map: &HashMap<NodeId, VariableId>,
    ) -> Self {
        let region_sources = source_nodes(compile_graph)
            .into_iter()
            .map(|wire| NodeId(mapped_wire(wire, wire_map)))
            .collect::<HashSet<_>>();
        let region_targets = target_nodes(compile_graph)
            .into_iter()
            .map(|wire| NodeId(mapped_wire(wire, wire_map)))
            .collect::<HashSet<_>>();
        let mut all = region_sources.clone();
        all.extend(region_targets.iter().copied());

        let mut control_sources_by_boundary_wire = HashMap::new();
        let mut control_targets_by_boundary_wire = HashMap::new();

        for operation_id in control_operation_ids {
            for wire in &operation_instances[*operation_id].outputs {
                let wire = NodeId(*wire);
                all.insert(wire);
                control_sources_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation_id);
            }

            for wire in &operation_instances[*operation_id].inputs {
                let wire = NodeId(*wire);
                all.insert(wire);
                control_targets_by_boundary_wire
                    .entry(wire)
                    .or_insert_with(Vec::new)
                    .push(*operation_id);
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

pub(super) fn entries_for_node(
    compile_graph: &CompileGraph,
    operations: &[OperationInstance],
    boundary: &BoundaryWires,
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

pub(super) fn exits_for_node(
    compile_graph: &CompileGraph,
    operations: &[OperationInstance],
    boundary: &BoundaryWires,
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
pub(super) fn boundary_point(
    compile_graph: &CompileGraph,
    wire: NodeId,
    kind: BoundaryKind,
) -> BoundaryPoint {
    BoundaryPoint {
        wire: wire.0,
        name: compile_graph.source_variable_names.get(&wire.0).cloned(),
        kind,
    }
}

pub(super) fn push_unique_boundary(target: &mut Vec<BoundaryPoint>, point: BoundaryPoint) {
    if !target.iter().any(|existing| existing == &point) {
        target.push(point);
    }
}

pub(super) fn push_unique_edge(target: &mut Vec<CfgEdge>, edge: CfgEdge) {
    if !target
        .iter()
        .any(|existing| existing.target == edge.target && existing.args == edge.args)
    {
        target.push(edge);
    }
}

pub(super) fn nodes_with_boundary(wiring: &CfgWiring, kind: BoundaryKind) -> Vec<CfgNodeId> {
    wiring
        .node_boundaries
        .iter()
        .filter(|boundaries| boundaries.entries.iter().any(|point| point.kind == kind))
        .map(|boundaries| boundaries.node)
        .collect()
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
