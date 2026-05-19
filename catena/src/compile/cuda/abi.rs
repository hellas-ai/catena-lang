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
        let dimension_leaf_names = dimension_leaf_names(&params);

        let mut used_param_names = HashSet::new();
        let mut unnamed_extent_count = 0usize;
        let mut global_count = 0usize;
        let mut params = Vec::new();
        let mut prelude = Vec::new();
        let mut names = HashMap::new();
        let mut extent_param_names = HashMap::new();
        let mut launch_block_size = None;
        let mut launch_element_count = None;
        let mut grid_shape = None;

        for id in &definition.params {
            let Some(variable) = definition.context.variable(*id) else {
                continue;
            };
            let mut alias = value_name(
                &variable.ty,
                &dimension_leaf_names,
                &mut unnamed_extent_count,
                &mut global_count,
            );
            if should_use_source_extent_name(&variable.ty, &alias, &dimension_leaf_names) {
                alias = sanitize_ident(&variable.name);
            }

            if let Some(param_ty) = cuda_param_type(&variable.ty) {
                let name = unique_name(&alias, &mut used_param_names);
                names.insert(variable.name.clone(), name.clone());
                if let Some(leaf) = extent_leaf(&variable.ty) {
                    extent_param_names.insert(leaf, name.clone());
                }
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
            } else if let Some(gpu_grid) = gpu_grid(&variable.ty) {
                let grid = dimension_leaves(gpu_grid.grid);
                let block = dimension_leaves(gpu_grid.block);
                if let (Some(grid), Some(block)) = (grid, block) {
                    grid_shape = Some(GridShape { grid, block });
                }
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
            launch: launch_config(
                grid_shape.and_then(|shape| resolve_grid_launch(&shape, &extent_param_names)),
                launch_block_size,
                launch_element_count,
            ),
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

#[derive(Debug, Clone)]
struct GridLaunch {
    grid: Vec<String>,
    block: Vec<String>,
}

#[derive(Debug, Clone)]
struct GridShape {
    grid: Vec<usize>,
    block: Vec<usize>,
}

fn resolve_grid_launch(
    shape: &GridShape,
    extent_param_names: &HashMap<usize, String>,
) -> Option<GridLaunch> {
    Some(GridLaunch {
        grid: resolve_dimension_names(&shape.grid, extent_param_names)?,
        block: resolve_dimension_names(&shape.block, extent_param_names)?,
    })
}

fn resolve_dimension_names(
    leaves: &[usize],
    extent_param_names: &HashMap<usize, String>,
) -> Option<Vec<String>> {
    leaves
        .iter()
        .map(|leaf| extent_param_names.get(leaf).cloned())
        .collect()
}

fn launch_config(
    grid: Option<GridLaunch>,
    block_size: Option<String>,
    element_count: Option<String>,
) -> CudaLaunch {
    if let Some(grid) = grid {
        if !grid.grid.is_empty() && !grid.block.is_empty() {
            let grid_expr = cuda_dim3_expr(&grid.grid);
            let block_expr = cuda_dim3_expr(&grid.block);
            let element_count = grid
                .grid
                .iter()
                .chain(grid.block.iter())
                .cloned()
                .collect::<Vec<_>>()
                .join(" * ");
            return CudaLaunch {
                block_expr,
                grid_expr,
                element_count: Some(element_count),
            };
        }
    }

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

fn cuda_dim3_expr(dimensions: &[String]) -> String {
    dimensions.join(", ")
}

fn dimension_leaf_names(variables: &[&Variable]) -> HashMap<usize, String> {
    let mut names = HashMap::new();
    let dimension_names = ["n", "m", "k"];

    for variable in variables {
        if let Some(global) = gpu_global(&variable.ty) {
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
    }

    names
}

fn dimension_leaves(dimension: &Obj) -> Option<Vec<usize>> {
    let Tree::Node(op, 0, children) = dimension else {
        return None;
    };
    let expected = match op.to_string().as_str() {
        "1d" => 1,
        "2d" => 2,
        "3d" => 3,
        _ => return None,
    };
    if children.len() != expected {
        return None;
    };
    children
        .iter()
        .map(|child| match child {
            Tree::Leaf(leaf, _) => Some(*leaf),
            _ => None,
        })
        .collect()
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

fn should_use_source_extent_name(
    obj: &Obj,
    alias: &str,
    dimension_leaf_names: &HashMap<usize, String>,
) -> bool {
    if extent_leaf(obj).is_none() {
        return false;
    }
    alias.starts_with("extent")
        || alias == "block_size"
            && !dimension_leaf_names
                .values()
                .any(|name| name == "block_size")
}

#[derive(Debug, Clone)]
struct GpuGrid<'a> {
    grid: &'a Obj,
    block: &'a Obj,
}

fn gpu_grid(obj: &Obj) -> Option<GpuGrid<'_>> {
    let obj = unwrap_val(obj).unwrap_or(obj);
    let Tree::Node(grid, 0, children) = obj else {
        return None;
    };
    if grid.to_string() != "gpu.grid" {
        return None;
    }
    let [grid, block] = children.as_slice() else {
        return None;
    };
    Some(GpuGrid { grid, block })
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

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
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
    dimensions: Vec<&'a Obj>,
}

fn gpu_global(obj: &Obj) -> Option<GpuGlobal<'_>> {
    let global = unwrap_val(obj)?;
    let Tree::Node(global, 0, children) = global else {
        return None;
    };
    let global_name = global.to_string();

    if global_name == "gpu.global" {
        let [Tree::Node(element, 0, _), dimensions] = children.as_slice() else {
            return None;
        };
        return Some(GpuGlobal {
            element: element.as_str(),
            dimensions: global_dimensions(dimensions)?,
        });
    };

    if matches!(
        global_name.as_str(),
        "gpu.global.1d" | "gpu.global.2d" | "gpu.global.3d"
    ) {
        let Some(Tree::Node(element, 0, _)) = children.first() else {
            return None;
        };
        return Some(GpuGlobal {
            element: element.as_str(),
            dimensions: children[1..].iter().collect(),
        });
    }

    None
}

fn global_dimensions(dimensions: &Obj) -> Option<Vec<&Obj>> {
    let Tree::Node(rank, 0, children) = dimensions else {
        return None;
    };
    let expected = match rank.to_string().as_str() {
        "1d" => 1,
        "2d" => 2,
        "3d" => 3,
        _ => return None,
    };
    if children.len() != expected {
        return None;
    }
    Some(children.iter().collect())
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
