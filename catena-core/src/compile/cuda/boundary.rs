use std::collections::{HashMap, HashSet};

use metacat::tree::Tree;

use crate::{
    compile::{
        cuda::{
            CudaOptions,
            abi::{CudaAbiError, CudaMacro},
            shape::{extent_leaf, memory_dimensions, rank_terms, unwrap_val},
            util::{macro_ident, sanitize_ident, unique_name},
        },
        program::Variable,
    },
    lang::Obj,
};

pub(super) struct KernelInterface {
    pub(super) grid_shape: GridShape,
    pub(super) extent_cuda_names: HashMap<usize, String>,
    pub(super) compile_time_extent_leaves: HashSet<usize>,
    pub(super) reserved_host_names: HashSet<String>,
    pub(super) macros: Vec<CudaMacro>,
}

#[derive(Debug, Clone)]
pub(super) struct GridShape {
    pub(super) grid: Vec<Obj>,
    pub(super) block: Vec<Obj>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuGlobal<'a> {
    pub(super) element: &'a str,
    pub(super) dimensions: Vec<&'a Obj>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuShared<'a> {
    pub(super) element: &'a str,
    pub(super) dimensions: Vec<&'a Obj>,
}

#[derive(Debug, Clone)]
pub(super) struct GpuGrid<'a> {
    grid: &'a Obj,
    block: &'a Obj,
}

impl GridShape {
    fn from_type(obj: &Obj) -> Result<Option<Self>, CudaAbiError> {
        let Some(gpu_grid) = gpu_grid(obj)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            grid: rank_terms(gpu_grid.grid).ok_or(CudaAbiError::InvalidGridShape)?,
            block: rank_terms(gpu_grid.block).ok_or(CudaAbiError::InvalidGridShape)?,
        }))
    }
}

pub(super) fn discover_kernel_interface(
    source_params: &[&Variable],
    options: &CudaOptions,
) -> Result<KernelInterface, CudaAbiError> {
    let mut grid_shape = None;
    let mut extent_cuda_names = HashMap::new();
    let mut compile_time_extent_leaves = HashSet::new();
    let mut reserved_host_names = HashSet::new();
    let mut used_macro_names = HashSet::new();
    let mut macros = Vec::new();
    let mut seen_static_names = HashSet::new();

    // This pass reads the source parameters of the entry arrow as the kernel
    // interface. Memory/grid dimensions can reference extent leaves that appear
    // elsewhere in the source object, and --cuda-static changes how those leaves
    // are named.
    for source_param in source_params {
        if let Some(shape) = GridShape::from_type(&source_param.ty)?
            && grid_shape.replace(shape).is_some()
        {
            return Err(CudaAbiError::DuplicateGrid);
        }

        let requested_static = options.static_values.get(&source_param.name).copied();
        if requested_static.is_some() {
            seen_static_names.insert(source_param.name.clone());
        }

        if let Some(leaf) = extent_leaf(&source_param.ty) {
            let name = if let Some(value) = requested_static {
                let name = unique_name(&macro_ident(&source_param.name), &mut used_macro_names);
                macros.push(CudaMacro {
                    name: name.clone(),
                    value,
                });
                compile_time_extent_leaves.insert(leaf);
                name
            } else {
                unique_name(
                    &sanitize_ident(&source_param.name),
                    &mut reserved_host_names,
                )
            };
            extent_cuda_names.insert(leaf, name);
        } else if requested_static.is_some() {
            return Err(CudaAbiError::StaticValueNotExtent(
                source_param.name.clone(),
            ));
        }
    }

    for name in options.static_values.keys() {
        if !seen_static_names.contains(name) {
            return Err(CudaAbiError::UnknownStaticValue(name.clone()));
        }
    }

    Ok(KernelInterface {
        grid_shape: grid_shape.ok_or(CudaAbiError::MissingGrid)?,
        extent_cuda_names,
        compile_time_extent_leaves,
        reserved_host_names,
        macros,
    })
}

pub(super) fn gpu_grid(obj: &Obj) -> Result<Option<GpuGrid<'_>>, CudaAbiError> {
    let Some(obj) = unwrap_val(obj) else {
        return Ok(None);
    };
    let Tree::Node(grid, 0, children) = obj else {
        return Ok(None);
    };
    if grid.to_string() != "gpu.grid" {
        return Ok(None);
    }
    let [grid, block] = children.as_slice() else {
        return Err(CudaAbiError::InvalidGridShape);
    };
    Ok(Some(GpuGrid { grid, block }))
}

pub(super) fn gpu_global(obj: &Obj) -> Result<Option<GpuGlobal<'_>>, CudaAbiError> {
    let Some((element, dimensions)) =
        gpu_memory(obj, "gpu.global", CudaAbiError::InvalidGlobalShape)?
    else {
        return Ok(None);
    };
    Ok(Some(GpuGlobal {
        element,
        dimensions,
    }))
}

pub(super) fn gpu_shared(obj: &Obj) -> Result<Option<GpuShared<'_>>, CudaAbiError> {
    let Some((element, dimensions)) =
        gpu_memory(obj, "gpu.shared", CudaAbiError::InvalidSharedShape)?
    else {
        return Ok(None);
    };
    Ok(Some(GpuShared {
        element,
        dimensions,
    }))
}

fn gpu_memory<'a>(
    obj: &'a Obj,
    expected_name: &str,
    invalid_shape: CudaAbiError,
) -> Result<Option<(&'a str, Vec<&'a Obj>)>, CudaAbiError> {
    let Some(memory) = unwrap_val(obj) else {
        return Ok(None);
    };
    let Tree::Node(memory, 0, children) = memory else {
        return Ok(None);
    };
    if memory.to_string() != expected_name {
        return Ok(None);
    }
    let [Tree::Node(element, 0, _), dimensions] = children.as_slice() else {
        return Err(invalid_shape);
    };
    let dimensions = memory_dimensions(dimensions).ok_or(invalid_shape)?;

    Ok(Some((element.as_str(), dimensions)))
}
