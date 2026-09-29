use std::collections::HashMap;

use crate::{
    codegen::{GpuModuleMap, lower_types::CType},
    runtime::{GeneratedFunction, value::ValueKind},
};

#[derive(Debug, Clone)]
pub(super) struct FunctionSignature {
    pub(super) symbol: String,
    pub(super) inputs: Vec<ValueKind>,
    pub(super) outputs: Vec<ValueKind>,
}

/// Source-level program names and their generated C ABI signatures.
pub(super) type SignatureTable = HashMap<String, FunctionSignature>;

pub(super) fn generated_signatures(
    functions: impl IntoIterator<Item = GeneratedFunction>,
) -> SignatureTable {
    functions
        .into_iter()
        .map(|function| {
            (
                function.source_name,
                FunctionSignature {
                    symbol: function.symbol,
                    inputs: function.inputs,
                    outputs: function.outputs,
                },
            )
        })
        .collect()
}

pub(super) fn generated_functions(modules: &GpuModuleMap) -> Vec<GeneratedFunction> {
    modules
        .values()
        .filter_map(|module| {
            let source_name = module.source_name.as_ref()?;
            let inputs = module
                .entry
                .sources
                .iter()
                .map(|var| {
                    let ty = crate::codegen::runtime_type(var)
                        .expect("GpuFunction sources should be runtime-lowered");
                    value_kind(ty)
                })
                .collect::<Option<Vec<_>>>()?;
            let outputs = module
                .entry
                .targets
                .iter()
                .map(|var| {
                    let ty = crate::codegen::runtime_type(var)
                        .expect("GpuFunction targets should be runtime-lowered");
                    value_kind(ty)
                })
                .collect::<Option<Vec<_>>>()?;

            Some(GeneratedFunction {
                source_name: source_name.to_string(),
                symbol: module.entry.name.clone(),
                inputs,
                outputs,
            })
        })
        .collect()
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
