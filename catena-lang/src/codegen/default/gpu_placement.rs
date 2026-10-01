use std::collections::{BTreeMap, HashSet};

use crate::codegen::default::{
    GpuFunction, GpuModuleMap, GpuValue, components::value_expr, ops::optimized::OptimizedOperation,
};

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
        OptimizedOperation::from_operation(assignment.op.as_str()).is_some()
            || matches!(
                assignment.op.as_str(),
                "gpu.materialize"
                    | "materializec"
                    | "materializec.into"
                    | "materializec.borrow"
                    | "materializec.borrow-argmax-f32"
                    | "materializec.borrow-argmax-bf16"
                    | "materializec.borrow-topk-f32"
                    | "materializec.borrow-reduce-f32"
                    | "materializec.reduce-f32"
                    | "materializec.reduce-f32-pair"
                    | "materializec.softmax-f32"
                    | "buf.free"
            )
    })
}

fn callers_by_callee(modules: &GpuModuleMap) -> BTreeMap<String, Vec<String>> {
    let mut callers = BTreeMap::<String, Vec<String>>::new();
    for module in modules.values() {
        for assignment in &module.entry.assignments {
            if let Some(callee) = assignment.call_symbol.as_deref() {
                callers
                    .entry(callee.to_owned())
                    .or_default()
                    .push(module.entry.name.clone());
            }
            // Converted control-flow closures call their named functions even
            // though these edges have no direct call_symbol. Device compilation
            // must exclude callers of host-only branch/fold bodies as well.
            if matches!(assignment.op.as_str(), "bool.ifc" | "reducec") {
                for input in &assignment.inputs {
                    if matches!(input, GpuValue::FnSymbol(_)) {
                        callers
                            .entry(value_expr(input))
                            .or_default()
                            .push(module.entry.name.clone());
                    }
                }
            }
        }
    }
    callers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{GpuAssign, GpuModule, fn_ptrs::FnPtrSymbol};
    fn module(name: &str, op: &str, call: Option<&str>, inputs: Vec<GpuValue>) -> GpuModule {
        GpuModule {
            name: name.into(),
            source_name: None,
            entry: GpuFunction {
                name: name.into(),
                sources: vec![],
                targets: vec![],
                assignments: vec![GpuAssign {
                    op: op.parse().unwrap(),
                    input_sizes: vec![],
                    output_sizes: vec![],
                    call_symbol: call.map(str::to_owned),
                    inputs,
                    outputs: vec![],
                }],
            },
        }
    }
    #[test]
    fn named_control_flow_closures_propagate_host_placement() {
        for op in ["bool.ifc", "reducec"] {
            let callee = module("program_host_callee", "buf.free", None, vec![]);
            let caller = module(
                "program_caller",
                op,
                None,
                vec![GpuValue::FnSymbol(FnPtrSymbol {
                    target: "host.callee".parse().unwrap(),
                })],
            );
            let outer = module("program_outer", "caller", Some("program_caller"), vec![]);
            let ordinary = module("program_ordinary", "u64.add", None, vec![]);
            let modules = [callee, caller, outer, ordinary]
                .into_iter()
                .map(|m| (m.name.parse().unwrap(), m))
                .collect();
            let placements = function_placements(&modules);
            for name in ["program_host_callee", "program_caller", "program_outer"] {
                assert!(placements[name].is_host_only(), "{op}: {name}");
            }
            assert!(!placements["program_ordinary"].is_host_only());
        }
    }
}
