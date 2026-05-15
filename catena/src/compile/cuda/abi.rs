use std::collections::{HashMap, HashSet};

use metacat::tree::Tree;

use crate::{
    compile::program::{Definition, Variable},
    lang::Obj,
    structured::ir::Param,
};

#[derive(Debug, Clone)]
pub(super) struct CudaKernelAbi {
    pub(super) params: Vec<Param>,
    pub(super) prelude: Vec<String>,
    pub(super) launch: CudaLaunch,
    names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct CudaLaunch {
    pub(super) block_expr: String,
    pub(super) grid_expr: String,
    pub(super) element_count: Option<String>,
}

impl CudaKernelAbi {
    pub(super) fn from_definition(definition: &Definition) -> Self {
        let params = definition
            .params
            .iter()
            .filter_map(|id| definition.context.variable(*id))
            .collect::<Vec<_>>();
        let dimension_leaf_names = global_dimension_leaf_names(&params);

        let mut used_param_names = HashSet::new();
        let mut unnamed_extent_count = 0usize;
        let mut global_count = 0usize;
        let mut params = Vec::new();
        let mut prelude = Vec::new();
        let mut names = HashMap::new();
        let mut launch_block_size = None;
        let mut launch_element_count = None;

        for id in &definition.params {
            let Some(variable) = definition.context.variable(*id) else {
                continue;
            };
            let alias = value_name(
                &variable.ty,
                &dimension_leaf_names,
                &mut unnamed_extent_count,
                &mut global_count,
            );

            if let Some(param_ty) = cuda_param_type(&variable.ty) {
                let name = unique_name(&alias, &mut used_param_names);
                names.insert(variable.name.clone(), name.clone());
                if extent_leaf(&variable.ty).is_some()
                    && !dimension_leaf_names
                        .values()
                        .any(|dim_name| dim_name == &name)
                    && launch_block_size.is_none()
                {
                    launch_block_size = Some(name.clone());
                }
                if launch_element_count.is_none()
                    && let Some(global) = gpu_global(&variable.ty)
                {
                    let dimensions = global
                        .dimensions
                        .iter()
                        .filter_map(|dimension| dimension_name(dimension, &dimension_leaf_names))
                        .collect::<Vec<_>>();
                    if !dimensions.is_empty() {
                        launch_element_count = Some(dimensions.join(" * "));
                    }
                }
                params.push(Param {
                    ty: param_ty.to_string(),
                    name,
                });
            } else if is_wrapped_type(&variable.ty, "gpu.block") {
                names.insert(variable.name.clone(), "block".to_string());
                prelude.push("uint3 block = blockIdx;".to_string());
            } else if is_wrapped_type(&variable.ty, "gpu.thread") {
                names.insert(variable.name.clone(), "thread".to_string());
                prelude.push("uint3 thread = threadIdx;".to_string());
            }
        }

        Self {
            params,
            prelude,
            launch: launch_config(launch_block_size, launch_element_count),
            names,
        }
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
}

fn launch_config(block_size: Option<String>, element_count: Option<String>) -> CudaLaunch {
    match (block_size, element_count) {
        (Some(block_size), Some(element_count)) => CudaLaunch {
            block_expr: block_size.clone(),
            grid_expr: format!("({element_count} + {block_size} - 1) / {block_size}"),
            element_count: Some(element_count),
        },
        _ => CudaLaunch {
            block_expr: "1".to_string(),
            grid_expr: "1".to_string(),
            element_count: None,
        },
    }
}

fn global_dimension_leaf_names(variables: &[&Variable]) -> HashMap<usize, String> {
    let mut names = HashMap::new();
    let dimension_names = ["n", "m", "k"];

    for variable in variables {
        let Some(global) = gpu_global(&variable.ty) else {
            continue;
        };
        for (index, dim) in global.dimensions.iter().enumerate() {
            if let Tree::Leaf(leaf, _) = dim {
                let name = dimension_names
                    .get(index)
                    .map(|name| (*name).to_string())
                    .unwrap_or_else(|| format!("dim{index}"));
                names.entry(*leaf).or_insert(name);
            }
        }
    }

    names
}

fn dimension_name(
    dimension: &Obj,
    dimension_leaf_names: &HashMap<usize, String>,
) -> Option<String> {
    let Tree::Leaf(leaf, _) = dimension else {
        return None;
    };
    dimension_leaf_names.get(leaf).cloned()
}

fn value_name(
    obj: &Obj,
    dimension_leaf_names: &HashMap<usize, String>,
    unnamed_extent_count: &mut usize,
    global_count: &mut usize,
) -> String {
    if let Some(leaf) = extent_leaf(obj) {
        if let Some(name) = dimension_leaf_names.get(&leaf) {
            return name.clone();
        }
        let name = if *unnamed_extent_count == 0 {
            "block_size".to_string()
        } else {
            format!("extent{}", unnamed_extent_count)
        };
        *unnamed_extent_count += 1;
        return name;
    }

    if is_wrapped_type(obj, "gpu.block") {
        return "block".to_string();
    }
    if is_wrapped_type(obj, "gpu.thread") {
        return "thread".to_string();
    }
    if gpu_global(obj).is_some() {
        let name = if *global_count == 0 {
            "out".to_string()
        } else {
            format!("global{}", global_count)
        };
        *global_count += 1;
        return name;
    }

    String::new()
}

fn unique_name(name: &str, used_names: &mut HashSet<String>) -> String {
    let name = if name.is_empty() { "param" } else { name };
    if used_names.insert(name.to_string()) {
        return name.to_string();
    }
    for suffix in 1.. {
        let candidate = format!("{name}{suffix}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search should always return")
}

fn cuda_param_type(obj: &Obj) -> Option<&'static str> {
    if extent_leaf(obj).is_some() {
        return Some("uint64_t");
    }
    let global = gpu_global(obj)?;
    match global.element {
        "f32" => Some("float*"),
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct GpuGlobal<'a> {
    element: &'a str,
    dimensions: &'a [Obj],
}

fn gpu_global(obj: &Obj) -> Option<GpuGlobal<'_>> {
    let global = unwrap_val(obj)?;
    let Tree::Node(global, 0, children) = global else {
        return None;
    };
    let global_name = global.to_string();
    if !matches!(
        global_name.as_str(),
        "gpu.global.1d" | "gpu.global.2d" | "gpu.global.3d"
    ) {
        return None;
    }
    let Some(Tree::Node(element, 0, _)) = children.first() else {
        return None;
    };
    Some(GpuGlobal {
        element: element.as_str(),
        dimensions: &children[1..],
    })
}

fn extent_leaf(obj: &Obj) -> Option<usize> {
    let extent = unwrap_val(obj)?;
    let Tree::Node(extent, 0, children) = extent else {
        return None;
    };
    if extent.to_string() != "extent" {
        return None;
    }
    let [Tree::Leaf(leaf, _)] = children.as_slice() else {
        return None;
    };
    Some(*leaf)
}

fn is_wrapped_type(obj: &Obj, type_name: &str) -> bool {
    matches!(
        unwrap_val(obj),
        Some(Tree::Node(label, 0, _)) if label.to_string() == type_name
    )
}

fn unwrap_val(obj: &Obj) -> Option<&Obj> {
    match obj {
        Tree::Node(wrapper, 0, children) if wrapper.to_string() == "val" => {
            let [inner] = children.as_slice() else {
                return None;
            };
            Some(inner)
        }
        _ => None,
    }
}
