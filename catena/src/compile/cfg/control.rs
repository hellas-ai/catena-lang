use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::compile::{CompileGraph, CompileTheory};

use super::{
    build::{CfgBuilder, OperationIdAllocator, VariableIdAllocator},
    model::{Cfg, CfgError, CfgOptions, OperationId, VariableId},
    monoidal::{MonoidalStructureResolver, MonoidalStructureSubgraph},
    operation::{
        OperationInstance, all_operation_wires, child_data_graph_for_operation,
        is_branch_operation, next_variable_id, operation_names, operation_sources,
        operation_targets, source_nodes, target_nodes,
    },
};

// Control expansion

#[derive(Debug, Clone)]
pub(super) struct ExpandedControlGraph {
    pub(super) items: Vec<ExpandedControlItem>,
    pub(super) visible_operation_to_entry: HashMap<OperationId, OperationId>,
}

#[derive(Debug, Clone)]
pub(super) enum ExpandedControlItem {
    Control(OperationInstance),
    DataCfg { call: OperationInstance, cfg: Cfg },
}

#[derive(Debug)]
pub(super) struct ControlExpander<'a> {
    compile_graph: &'a CompileGraph,
    operation_instances: &'a [OperationInstance],
    monoidal_structure_subgraph: MonoidalStructureSubgraph,
    current_branch: Option<OperationInstance>,
    branch_data_successor_counts: HashMap<OperationId, usize>,
    branch_payload_by_operation: HashMap<OperationId, VariableId>,
    operation_ids: OperationIdAllocator,
    variable_ids: VariableIdAllocator,
    options: CfgOptions,
}

impl<'a> ControlExpander<'a> {
    pub(super) fn new(
        compile_graph: &'a CompileGraph,
        operation_instances: &'a [OperationInstance],
        monoidal_structure_subgraph: MonoidalStructureSubgraph,
        options: CfgOptions,
    ) -> Self {
        Self {
            compile_graph,
            operation_instances,
            monoidal_structure_subgraph,
            current_branch: None,
            branch_data_successor_counts: HashMap::new(),
            branch_payload_by_operation: HashMap::new(),
            operation_ids: OperationIdAllocator::new(operation_instances.len()),
            variable_ids: VariableIdAllocator::new(next_variable_id(operation_instances)),
            options,
        }
    }

    pub(super) fn expand(
        mut self,
        control_operation_ids: &[OperationId],
    ) -> Result<ExpandedControlGraph, CfgError> {
        let mut items = Vec::new();
        let mut visible_operation_to_entry = HashMap::new();

        for operation_id in control_operation_ids {
            let call = &self.operation_instances[*operation_id];
            let first = items.len();
            let monoidal_structure_subgraph = self.monoidal_structure_subgraph.clone();
            self.inline_operation(
                self.compile_graph,
                call,
                true,
                &monoidal_structure_subgraph,
                &mut items,
            )?;
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
        monoidal_structure_subgraph: &MonoidalStructureSubgraph,
        output: &mut Vec<ExpandedControlItem>,
    ) -> Result<(), CfgError> {
        let Some(child) = self.child_control_graph(compile_graph, &call.name) else {
            if let Some(child) = child_data_graph_for_operation(compile_graph, &call.name) {
                let call = self.call_with_branch_payload(call, monoidal_structure_subgraph);
                let cfg = self.remapped_data_cfg(child, &call, monoidal_structure_subgraph)?;
                output.push(ExpandedControlItem::DataCfg { call, cfg });
                return Ok(());
            }
            let mut operation = call.clone();
            if !keep_call_id {
                operation.id = self.operation_ids.allocate();
            }
            self.current_branch = is_branch_operation(&operation).then_some(operation.clone());
            output.push(ExpandedControlItem::Control(operation));
            return Ok(());
        };

        let wire_map = self.child_wire_map(child, call);
        let child_monoidal_structure_subgraph =
            MonoidalStructureSubgraph::from_compile_graph_with_context(
                child,
                Some(&wire_map),
                Some(monoidal_structure_subgraph),
            );
        let child_monoidal_structure_resolver =
            MonoidalStructureResolver::from_subgraph(child, child_monoidal_structure_subgraph);
        for child_operation_id in 0..operation_names(child).len() {
            let child_call = self.remapped_child_operation(
                child,
                child_operation_id,
                &wire_map,
                &child_monoidal_structure_resolver,
            )?;
            self.inline_operation(
                child,
                &child_call,
                false,
                child_monoidal_structure_resolver.subgraph(),
                output,
            )?;
        }
        Ok(())
    }

    fn call_with_branch_payload(
        &mut self,
        call: &OperationInstance,
        monoidal_structure_subgraph: &MonoidalStructureSubgraph,
    ) -> OperationInstance {
        let Some(branch) = self.current_branch.as_ref() else {
            return call.clone();
        };
        let branch_index = self
            .branch_data_successor_counts
            .entry(branch.id)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        let Some(payload) = branch.outputs.get(*branch_index - 1).copied() else {
            return call.clone();
        };
        let monoidal_structure_resolver = MonoidalStructureResolver::from_subgraph(
            self.compile_graph,
            monoidal_structure_subgraph.clone(),
        );
        let payload = monoidal_structure_resolver.resolve_branch_payload_wire(payload);
        let payload = if payload == branch.outputs[*branch_index - 1] {
            self.branch_payload_by_operation
                .get(&branch.id)
                .copied()
                .unwrap_or(payload)
        } else {
            self.branch_payload_by_operation.insert(branch.id, payload);
            payload
        };
        let mut call = call.clone();
        if let Some(input) = call.inputs.first_mut() {
            *input = payload;
        }
        call
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
        monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    ) -> Result<OperationInstance, CfgError> {
        let mut operation = OperationInstance {
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
            branch_condition: None,
        };
        if is_branch_operation(&operation)
            && let Some(condition) = operation.inputs.first().copied()
        {
            operation.branch_condition =
                Some(monoidal_structure_resolver.resolve_discriminator(condition)?);
        }
        Ok(operation)
    }

    fn remapped_data_cfg(
        &mut self,
        child: &CompileGraph,
        call: &OperationInstance,
        monoidal_structure_subgraph: &MonoidalStructureSubgraph,
    ) -> Result<Cfg, CfgError> {
        let variable_map = self.child_wire_map(child, call);
        CfgBuilder::new_with_context_and_monoidal(
            child,
            variable_map,
            Some(monoidal_structure_subgraph.clone()),
        )
        .with_options(self.options)
        .build()
    }
}
