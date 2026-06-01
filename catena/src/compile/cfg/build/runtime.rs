use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::compile::cfg::model::{CfgError, OperationId, VariableId};
use crate::compile::{CompileGraph, CompileTheory};

use super::{
    monoidal::MonoidalStructureResolver,
    operation::{
        CfgOperationRole, OperationInstance, cfg_operation_role, effective_operation_instance,
        is_control_operation, operation_names,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeOperationRole {
    Data,
    Control,
    Skip,
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeGraph {
    pub(super) operations: Vec<OperationInstance>,
    pub(super) control_ids: Vec<OperationId>,
    pub(super) data_ids: Vec<OperationId>,
}

impl RuntimeGraph {
    pub(super) fn collect(
        compile_graph: &CompileGraph,
        wire_map: &HashMap<NodeId, VariableId>,
        monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    ) -> Result<Self, CfgError> {
        let operations = (0..operation_names(compile_graph).len())
            .map(|operation_id| {
                effective_operation_instance(
                    compile_graph,
                    operation_id,
                    wire_map,
                    monoidal_structure_resolver,
                )
            })
            .collect::<Result<Vec<_>, CfgError>>()?;

        let mut control_ids = Vec::new();
        let mut data_ids = Vec::new();
        for operation in &operations {
            match cfg_runtime_role(compile_graph, operation) {
                RuntimeOperationRole::Control => control_ids.push(operation.id),
                RuntimeOperationRole::Data => data_ids.push(operation.id),
                RuntimeOperationRole::Skip => {}
            }
        }

        Ok(Self {
            operations,
            control_ids,
            data_ids,
        })
    }
}

fn cfg_runtime_role(
    compile_graph: &CompileGraph,
    operation: &OperationInstance,
) -> RuntimeOperationRole {
    match cfg_operation_role(&operation.name) {
        CfgOperationRole::MonoidalStructure => RuntimeOperationRole::Skip,
        CfgOperationRole::ControlFlow => RuntimeOperationRole::Control,
        CfgOperationRole::Instruction if matches!(compile_graph.theory, CompileTheory::Control) => {
            RuntimeOperationRole::Control
        }
        CfgOperationRole::Instruction if is_control_operation(compile_graph, &operation.name) => {
            RuntimeOperationRole::Control
        }
        CfgOperationRole::Instruction => RuntimeOperationRole::Data,
    }
}
