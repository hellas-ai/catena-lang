use open_hypergraphs::lax::NodeId;
use std::collections::HashMap;

use crate::compile::{CompileGraph, CompileTheory};

use super::{
    model::{CfgError, CfgOptions, OperationId, OperationName, VariableId},
    monoidal::MonoidalStructureResolver,
};

pub(super) const MONOIDAL_STRUCTURE_OPERATIONS: &[&str] = &[
    "val.*.intro",
    "val.*.elim",
    "val.+.intro",
    "val.+.elim",
    "2.intro",
    "2.elim",
    "distl",
    "distr",
    "unitl.intro",
    "unitl.elim",
    "elim2",
];

pub(super) const CONTROL_FLOW_ONLY_OPERATIONS: &[&str] = &["merge", "never"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CfgOperationRole {
    Instruction,
    MonoidalStructure,
    ControlFlow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OperationInstance {
    pub id: OperationId,
    pub name: OperationName,
    pub inputs: Vec<VariableId>,
    pub outputs: Vec<VariableId>,
    pub branch_condition: Option<VariableId>,
}

pub(super) fn cfg_operation_role(operation: &str) -> CfgOperationRole {
    if operation.starts_with("control.") {
        return CfgOperationRole::ControlFlow;
    }

    let local = local_operation_name(operation);
    if CONTROL_FLOW_ONLY_OPERATIONS.contains(&local) {
        CfgOperationRole::ControlFlow
    } else if MONOIDAL_STRUCTURE_OPERATIONS.contains(&local) {
        CfgOperationRole::MonoidalStructure
    } else {
        CfgOperationRole::Instruction
    }
}

pub(super) fn local_operation_name(operation: &str) -> &str {
    operation
        .strip_prefix("data.")
        .or_else(|| operation.strip_prefix("control."))
        .unwrap_or(operation)
}

pub(super) fn child_data_graph_for_operation<'a>(
    compile_graph: &'a CompileGraph,
    operation: &str,
) -> Option<&'a CompileGraph> {
    let local_name = local_operation_name(operation);
    compile_graph
        .children
        .iter()
        .find(|child| {
            child.operation == operation
                || child.operation == local_name
                || child.graph.definition_name == local_name
        })
        .map(|child| &child.graph)
        .filter(|child| matches!(child.theory, CompileTheory::Data))
}

pub(super) fn is_branch_operation(operation: &OperationInstance) -> bool {
    operation.outputs.len() > 1 || operation.name.contains("branch") || operation.name == "if"
}

// Compile graph accessors

pub(super) fn operation_instance(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
) -> OperationInstance {
    OperationInstance {
        id: operation_id,
        name: operation_names(compile_graph)[operation_id].to_string(),
        inputs: variables(&operation_sources(compile_graph, operation_id)),
        outputs: variables(&operation_targets(compile_graph, operation_id)),
        branch_condition: None,
    }
}

pub(super) fn effective_operation_instance(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
    wire_map: &HashMap<NodeId, VariableId>,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    options: CfgOptions,
) -> Result<OperationInstance, CfgError> {
    let mut operation = operation_instance(compile_graph, operation_id);
    operation.inputs = operation
        .inputs
        .into_iter()
        .map(|wire| mapped_wire(NodeId(wire), wire_map))
        .collect();
    operation.branch_condition =
        resolve_branch_condition(compile_graph, &operation, monoidal_structure_resolver)?;
    operation.inputs = resolve_instruction_inputs(
        compile_graph,
        operation.clone(),
        monoidal_structure_resolver,
        options,
    )?;
    operation.outputs = operation
        .outputs
        .into_iter()
        .map(|wire| mapped_wire(NodeId(wire), wire_map))
        .collect();
    Ok(operation)
}

fn resolve_instruction_inputs(
    compile_graph: &CompileGraph,
    operation: OperationInstance,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    options: CfgOptions,
) -> Result<Vec<VariableId>, CfgError> {
    if !options.keep_monoidal_operations
        && !is_control_operation(compile_graph, &operation.name)
        && matches!(
            cfg_operation_role(&operation.name),
            CfgOperationRole::Instruction
        )
    {
        monoidal_structure_resolver.resolve_variables(operation.inputs)
    } else {
        Ok(operation.inputs)
    }
}

fn resolve_branch_condition(
    compile_graph: &CompileGraph,
    operation: &OperationInstance,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
) -> Result<Option<VariableId>, CfgError> {
    if !is_control_operation(compile_graph, &operation.name) || !is_branch_operation(operation) {
        return Ok(None);
    }
    operation
        .inputs
        .first()
        .copied()
        .map(|input| monoidal_structure_resolver.resolve_discriminator(input))
        .transpose()
}

pub(super) fn mapped_wire(wire: NodeId, wire_map: &HashMap<NodeId, VariableId>) -> VariableId {
    wire_map.get(&wire).copied().unwrap_or(wire.0)
}

pub(super) fn next_variable_id(operation_instances: &[OperationInstance]) -> VariableId {
    operation_instances
        .iter()
        .flat_map(|operation| operation.inputs.iter().chain(&operation.outputs))
        .copied()
        .max()
        .map(|variable| variable + 1)
        .unwrap_or(0)
}

pub(super) fn is_control_operation(compile_graph: &CompileGraph, operation: &str) -> bool {
    operation.starts_with("control.")
        || compile_graph
            .children
            .iter()
            .find(|child| child.operation == operation)
            .map(|child| &child.graph)
            .is_some_and(|child| matches!(child.theory, CompileTheory::Control))
}

pub(super) fn source_nodes(compile_graph: &CompileGraph) -> Vec<NodeId> {
    compile_graph
        .graph
        .s
        .table
        .iter()
        .copied()
        .map(NodeId)
        .collect()
}

pub(super) fn target_nodes(compile_graph: &CompileGraph) -> Vec<NodeId> {
    compile_graph
        .graph
        .t
        .table
        .iter()
        .copied()
        .map(NodeId)
        .collect()
}

pub(super) fn operation_names(compile_graph: &CompileGraph) -> &[crate::lang::Arr] {
    &compile_graph.graph.h.x.0
}

pub(super) fn operation_sources(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
) -> Vec<NodeId> {
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

pub(super) fn operation_targets(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
) -> Vec<NodeId> {
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

pub(super) fn variables(nodes: &[NodeId]) -> Vec<VariableId> {
    nodes.iter().map(|node| node.0).collect()
}

pub(super) fn all_operation_wires(
    compile_graph: &CompileGraph,
    operation_id: OperationId,
) -> Vec<NodeId> {
    let mut wires = operation_sources(compile_graph, operation_id);
    wires.extend(operation_targets(compile_graph, operation_id));
    wires
}
