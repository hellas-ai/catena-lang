use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::compile::{CompileGraph, CompileTheory};

use super::{
    build::{CfgBuilder, OperationIdAllocator, VariableIdAllocator},
    model::{Cfg, CfgError, OperationId, VariableId},
    operation::{
        OperationInstance, all_operation_wires, child_data_graph_for_operation, next_variable_id,
        operation_names, operation_sources, operation_targets, source_nodes, target_nodes,
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
    operation_ids: OperationIdAllocator,
    variable_ids: VariableIdAllocator,
}

impl<'a> ControlExpander<'a> {
    pub(super) fn new(
        compile_graph: &'a CompileGraph,
        operation_instances: &'a [OperationInstance],
    ) -> Self {
        Self {
            compile_graph,
            operation_instances,
            operation_ids: OperationIdAllocator::new(operation_instances.len()),
            variable_ids: VariableIdAllocator::new(next_variable_id(operation_instances)),
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
        let variable_map = self.child_wire_map(child, call);
        CfgBuilder::new_with_context(child, variable_map).build()
    }
}
