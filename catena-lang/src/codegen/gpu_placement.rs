use std::collections::{BTreeMap, HashSet};

use crate::codegen::{GpuFunction, GpuModuleMap};

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
    function
        .assignments
        .iter()
        .any(|assignment| matches!(assignment.op.as_str(), "gpu.materialize" | "materializec"))
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
