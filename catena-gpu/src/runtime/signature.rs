use std::collections::HashMap;

use crate::{
    codegen::{GpuModuleMap, lower_types::CType, runtime_type},
    runtime::ValueKind,
};

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub symbol: String,
    pub inputs: Vec<ValueKind>,
    pub outputs: Vec<ValueKind>,
}

pub(crate) type SignatureTable = HashMap<String, FunctionSignature>;

pub(crate) fn signatures(modules: &GpuModuleMap) -> SignatureTable {
    modules
        .values()
        .filter_map(|module| {
            let source = module.source_name.as_ref()?;
            let inputs = module
                .entry
                .sources
                .iter()
                .map(|var| value_kind(runtime_type(var)?))
                .collect::<Option<Vec<_>>>()?;
            let outputs = module
                .entry
                .targets
                .iter()
                .map(|var| value_kind(runtime_type(var)?))
                .collect::<Option<Vec<_>>>()?;
            Some((
                source.to_string(),
                FunctionSignature {
                    symbol: module.name.clone(),
                    inputs,
                    outputs,
                },
            ))
        })
        .collect()
}

fn value_kind(ty: &CType) -> Option<ValueKind> {
    Some(match ty {
        CType::Bool => ValueKind::Bool,
        CType::U32 => ValueKind::U32,
        CType::U64 => ValueKind::U64,
        CType::F32 => ValueKind::F32,
    })
}
