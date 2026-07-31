use std::collections::{BTreeMap, HashSet};

use crate::codegen::{GpuFunction, GpuModuleMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GpuFunctionPlacement {
    HostOnly,
    DeviceOnly,
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
    } else if function_directly_requires_device(function) {
        GpuFunctionPlacement::DeviceOnly
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

    let mut device_only = modules
        .values()
        .filter(|module| function_directly_requires_device(&module.entry))
        .map(|module| module.entry.name.clone())
        .collect::<HashSet<_>>();
    let mut frontier = device_only.iter().cloned().collect::<Vec<_>>();

    while let Some(device_only_callee) = frontier.pop() {
        if let Some(callers) = callers_by_callee.get(device_only_callee.as_str()) {
            for caller in callers {
                if device_only.insert(caller.clone()) {
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
            } else if device_only.contains(&module.entry.name) {
                GpuFunctionPlacement::DeviceOnly
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
        .any(|assignment| assignment.op.as_str() == "gpu.materialize")
}

fn function_directly_requires_device(function: &GpuFunction) -> bool {
    function.assignments.iter().any(|assignment| {
        matches!(
            assignment.op.as_str(),
            "gpu.shared.alloc"
                | "gpu.shared.row-major.cooperative-loadc"
                | "gpu.shared.materialize"
                | "gpu.sync"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::GpuAssign;

    #[test]
    fn shared_memory_functions_are_device_only() {
        let function = GpuFunction {
            name: "shared_kernel_body".to_string(),
            sources: Vec::new(),
            targets: Vec::new(),
            assignments: vec![GpuAssign {
                op: "gpu.shared.alloc".parse().unwrap(),
                input_sizes: Vec::new(),
                output_sizes: Vec::new(),
                call_symbol: None,
                inputs: Vec::new(),
                outputs: Vec::new(),
            }],
        };

        assert_eq!(
            direct_function_placement(&function),
            GpuFunctionPlacement::DeviceOnly
        );
    }
}
