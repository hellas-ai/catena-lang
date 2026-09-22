use std::collections::{BTreeMap, HashSet};

use crate::codegen::default::{GpuFunction, GpuModuleMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuFunctionPlacement {
    HostOnly,
    HostAndDevice,
}

impl GpuFunctionPlacement {
    pub(super) fn is_host_only(self) -> bool {
        matches!(self, Self::HostOnly)
    }
}

#[cfg(test)]
pub(super) fn direct_function_placement(function: &GpuFunction) -> GpuFunctionPlacement {
    if function_directly_requires_host(function) {
        GpuFunctionPlacement::HostOnly
    } else {
        GpuFunctionPlacement::HostAndDevice
    }
}

pub(super) fn function_placements(
    modules: &GpuModuleMap,
) -> BTreeMap<String, GpuFunctionPlacement> {
    let callers_by_callee = callers_by_callee(modules);
    let mut host_only = modules
        .values()
        .filter(|module| function_directly_requires_host(&module.entry))
        .map(|module| module.entry.name.clone())
        .collect::<HashSet<_>>();
    let mut frontier = host_only.iter().cloned().collect::<Vec<_>>();

    while let Some(host_only_callee) = frontier.pop() {
        if let Some(callers) = callers_by_callee.get(host_only_callee.as_str()) {
            for caller in callers {
                if host_only.insert(caller.clone()) {
                    frontier.push(caller.clone());
                }
            }
        }
    }

    modules
        .values()
        .map(|module| {
            let placement = if host_only.contains(&module.entry.name) {
                GpuFunctionPlacement::HostOnly
            } else {
                GpuFunctionPlacement::HostAndDevice
            };
            (module.entry.name.clone(), placement)
        })
        .collect()
}

pub(super) fn function_placement(
    placements: &BTreeMap<String, GpuFunctionPlacement>,
    function_name: &str,
) -> GpuFunctionPlacement {
    placements
        .get(function_name)
        .copied()
        .unwrap_or(GpuFunctionPlacement::HostAndDevice)
}

fn function_directly_requires_host(function: &GpuFunction) -> bool {
    function.assignments.iter().any(|assignment| {
        matches!(
            assignment.op.as_str(),
            "gpu.materialize"
                | "materializec"
                | "materializec.into"
                | "materializec.borrow"
                | "materializec.borrow-argmax-f32"
                | "materializec.borrow-argmax-bf16"
                | "materializec.borrow-topk-f32"
                | "materializec.borrow-reduce-f32"
                | "materializec.borrow-cached-context-f32"
                | "materializec.borrow-cached-context-bf16"
                | "materializec.borrow-cached-attention-bf16"
                | "materializec.borrow-cached-attention-bf16-wmma"
                | "materializec.borrow-routed-bf16-gemv-pair"
                | "materializec.borrow-native-bf16-gemv-pair"
                | "materializec.borrow-native-bf16-gemm-wmma-pair"
                | "materializec.borrow-native-bf16-gemm-blas-pair"
                | "materializec.borrow-routed-native-bf16-gemv-pair"
                | "materializec.borrow-routed-native-bf16-gemm-wmma-pair"
                | "materializec.borrow-routed-native-bf16-gemv-residual"
                | "materializec.borrow-routed-native-bf16-gemm-wmma-residual"
                | "materializec.bf16-gemv"
                | "materializec.bf16-gemm-wmma"
                | "materializec.bf16-gemm-blas"
                | "materializec.borrow-gated-delta-net-f32"
                | "materializec.borrow-gated-delta-net-bf16"
                | "materializec.reduce-f32"
                | "materializec.reduce-f32-pair"
                | "materializec.softmax-f32"
                | "buf.free"
        )
    })
}

fn callers_by_callee(modules: &GpuModuleMap) -> BTreeMap<&str, Vec<String>> {
    let mut callers = BTreeMap::<&str, Vec<String>>::new();
    for module in modules.values() {
        for assignment in &module.entry.assignments {
            if let Some(callee) = assignment.call_symbol.as_deref() {
                callers
                    .entry(callee)
                    .or_default()
                    .push(module.entry.name.clone());
            }
        }
    }
    callers
}
