use std::collections::BTreeMap;

use hexpr::Operation;

use super::{GpuValue, GpuVar, lower_types::CType, runtime_type};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpecializationKey {
    pub sources: Vec<CType>,
    pub targets: Vec<CType>,
    pub static_inputs: Vec<Operation>,
}

pub struct PendingInstance {
    pub operation: Operation,
    pub symbol: String,
    pub source_name: Option<Operation>,
    pub substitutions: BTreeMap<usize, CType>,
}

pub fn key(inputs: &[GpuValue], outputs: &[GpuVar]) -> SpecializationKey {
    SpecializationKey {
        sources: inputs
            .iter()
            .filter_map(|input| match input {
                GpuValue::Var(var) => runtime_type(var).cloned(),
                GpuValue::FnSymbol(_) => None,
            })
            .collect(),
        targets: outputs
            .iter()
            .filter_map(|output| runtime_type(output).cloned())
            .collect(),
        static_inputs: inputs
            .iter()
            .filter_map(|input| match input {
                GpuValue::FnSymbol(symbol) => Some(symbol.clone()),
                GpuValue::Var(_) => None,
            })
            .collect(),
    }
}

pub fn boundary_key(sources: Vec<CType>, targets: Vec<CType>) -> SpecializationKey {
    SpecializationKey {
        sources,
        targets,
        static_inputs: Vec::new(),
    }
}
