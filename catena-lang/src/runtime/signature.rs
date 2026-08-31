use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    codegen::{GpuModuleMap, lower_types::CType},
    runtime::value::ValueKind,
};

/// Read-only source-level function signature exposed by a loaded artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    name: String,
    inputs: Vec<ValueKind>,
    outputs: Vec<ValueKind>,
}

impl EntryPoint {
    pub(crate) fn new(name: String, inputs: Vec<ValueKind>, outputs: Vec<ValueKind>) -> Self {
        Self {
            name,
            inputs,
            outputs,
        }
    }

    /// Source-level function name used for execution.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Ordered input kinds, including memory ownership.
    pub fn inputs(&self) -> &[ValueKind] {
        &self.inputs
    }

    /// Ordered output kinds, including memory ownership.
    pub fn outputs(&self) -> &[ValueKind] {
        &self.outputs
    }
}

#[derive(Debug, Clone)]
pub(super) struct FunctionSignature {
    pub(super) symbol: String,
    pub(super) inputs: Vec<ValueKind>,
    pub(super) outputs: Vec<ValueKind>,
}

/// Source-level program names and their generated C ABI signatures.
pub(super) type SignatureTable = HashMap<String, FunctionSignature>;

pub(super) fn entry_points(signatures: &SignatureTable) -> Vec<EntryPoint> {
    let mut entries = signatures
        .iter()
        .map(|(name, signature)| {
            EntryPoint::new(
                name.clone(),
                signature.inputs.clone(),
                signature.outputs.clone(),
            )
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    entries
}

pub(super) fn signatures(modules: &GpuModuleMap) -> SignatureTable {
    let mut signatures = HashMap::new();
    for module in modules.values() {
        let Some(source_name) = &module.source_name else {
            continue;
        };
        let Some(inputs) = module
            .entry
            .sources
            .iter()
            .map(|var| {
                let ty = crate::codegen::runtime_type(var)
                    .expect("GpuFunction sources should be runtime-lowered");
                value_kind(ty)
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(outputs) = module
            .entry
            .targets
            .iter()
            .map(|var| {
                let ty = crate::codegen::runtime_type(var)
                    .expect("GpuFunction targets should be runtime-lowered");
                value_kind(ty)
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };

        signatures.insert(
            source_name.to_string(),
            FunctionSignature {
                symbol: module.entry.name.clone(),
                inputs,
                outputs,
            },
        );
    }
    signatures
}

fn value_kind(ty: &CType) -> Option<ValueKind> {
    match ty {
        CType::Bool => Some(ValueKind::Bool),
        CType::U16 => Some(ValueKind::U16),
        CType::U32 => Some(ValueKind::U32),
        CType::U64 => Some(ValueKind::U64),
        CType::F32 => Some(ValueKind::F32),
        CType::Named(name) if name == "catena_mem_own_t" => Some(ValueKind::MemOwn),
        CType::Named(name) if name == "catena_mem_ref_t" => Some(ValueKind::MemRef),
        _ => None,
    }
}
