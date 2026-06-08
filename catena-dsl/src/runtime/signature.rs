use std::collections::HashMap;

use crate::{
    codegen::{GpuModuleMap, lower_types::CType},
    runtime::value::ValueKind,
};

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub(crate) symbol: String,
    pub(crate) inputs: Vec<ValueKind>,
    pub(crate) outputs: Vec<ValueKind>,
}

pub(crate) fn signatures(modules: &GpuModuleMap) -> HashMap<String, FunctionSignature> {
    let mut signatures = HashMap::new();
    for module in modules.values() {
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
            module.entry.name.clone(),
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
        CType::U64 => Some(ValueKind::U64),
        CType::Named(name) if name == "catena_mem_t" => Some(ValueKind::Mem),
        _ => None,
    }
}
