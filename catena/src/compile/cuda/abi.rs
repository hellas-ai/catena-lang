use std::collections::{HashMap, HashSet};

use metacat::tree::Tree;
use thiserror::Error;

use crate::{
    compile::program::{Definition, Variable},
    lang::Obj,
    structured::ir::Param,
};

#[derive(Debug, Clone)]
pub(super) struct CudaKernelAbi {
    pub(super) device_params: Vec<Param>,
    pub(super) host_params: Vec<Param>,
    pub(super) device_call_args: Vec<String>,
    pub(super) prelude: Vec<String>,
    pub(super) host_prelude: Vec<String>,
    pub(super) launch: CudaLaunch,
    names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct CudaLaunch {
    pub(super) block_expr: String,
    pub(super) grid_expr: String,
    pub(super) element_count: Option<String>,
}

#[derive(Debug, Error)]
pub enum CudaAbiError {
    #[error("definition parameter {0:?} is missing from its context")]
    MissingParamVariable(crate::compile::program::VariableId),
    #[error("CUDA kernel boundary is missing a gpu.grid value")]
    MissingGrid,
    #[error("CUDA kernel boundary must provide exactly one gpu.grid value")]
    DuplicateGrid,
    #[error("gpu.grid dimensions must be 1d, 2d, or 3d leaves backed by extent arguments")]
    InvalidGridShape,
    #[error("gpu.grid dimension leaf {0} is not backed by an extent argument")]
    MissingGridExtent(usize),
    #[error(
        "gpu.global boundary values must be gpu.global element dimensions with 1d, 2d, or 3d dimensions"
    )]
    InvalidGlobalShape,
    #[error("gpu.global dimension leaf {0} is not backed by an extent argument")]
    MissingGlobalExtent(usize),
    #[error("unsupported CUDA global memory element type `{0}`")]
    UnsupportedGlobalElement(String),
    #[error("unsupported CUDA kernel boundary argument `{name}` of type `{ty}`")]
    UnsupportedBoundaryArgument { name: String, ty: String },
}

impl CudaKernelAbi {
    pub(super) fn from_definition(definition: &Definition) -> Result<Self, CudaAbiError> {
        // We are compiling a Catena arrow as a CUDA kernel. Its boundary tells us
        // both the C ABI of the kernel and the launch shape.
        let boundary = definition
            .params
            .iter()
            .map(|id| {
                definition
                    .context
                    .variable(*id)
                    .ok_or(CudaAbiError::MissingParamVariable(*id))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let discovery = discover_boundary(&boundary)?;
        let mut used_device_names = HashSet::new();
        let mut used_host_names = discovery.used_host_names;
        let mut device_params = Vec::new();
        let mut host_params = Vec::new();
        let mut device_call_args = Vec::new();
        let mut host_prelude = Vec::new();
        let mut names = HashMap::new();
        let mut emitted_extent_params = HashSet::new();
        let mut element_count = None;

        // Emit the host launch ABI and the device kernel ABI in boundary order.
        // Extents stay on the host side; globals become device kernel pointers
        // plus size values derived from their gpu.global dimensions.
        for variable in boundary {
            if let Some(leaf) = extent_leaf(&variable.ty) {
                let Some(name) = discovery.extent_param_names.get(&leaf).cloned() else {
                    return Err(CudaAbiError::MissingGridExtent(leaf));
                };
                names.insert(variable.name.clone(), name.clone());
                if emitted_extent_params.insert(leaf) {
                    host_params.push(Param {
                        ty: "uint64_t".to_string(),
                        name,
                    });
                }
                continue;
            };

            if let Some(global) = gpu_global(&variable.ty)? {
                let param_ty = cuda_global_param_type(&global)?;
                let base_name = sanitize_ident(&variable.name);
                let device_name = unique_name(&base_name, &mut used_device_names);
                let host_name = unique_name(&base_name, &mut used_host_names);
                let size_name = unique_name(&format!("{device_name}_size"), &mut used_device_names);
                let size_expr = global_size_expr(&global, &discovery.extent_param_names)?;

                names.insert(variable.name.clone(), device_name.clone());
                device_params.push(Param {
                    ty: "uint64_t".to_string(),
                    name: size_name.clone(),
                });
                device_params.push(Param {
                    ty: param_ty.to_string(),
                    name: device_name,
                });
                host_params.push(Param {
                    ty: param_ty.to_string(),
                    name: host_name.clone(),
                });
                element_count.get_or_insert_with(|| size_name.clone());
                host_prelude.push(format!("uint64_t {size_name} = {size_expr};"));
                device_call_args.push(size_name);
                device_call_args.push(host_name);
                continue;
            };

            // gpu.grid values are handled above as launch contracts and erased
            // from the CUDA parameter list.
            if gpu_grid(&variable.ty)?.is_some() {
                continue;
            }
            //
            // TODO: support gpu.shared boundary values. Shared memory should be
            // emitted as kernel-local/shared declarations, not host ABI params.
            //
            // Other generic kernel arguments are intentionally unsupported in
            // this first CUDA path.
            return Err(CudaAbiError::UnsupportedBoundaryArgument {
                name: variable.name.clone(),
                ty: format!("{:?}", variable.ty),
            });
        }

        let launch = launch_config(
            resolve_grid_launch(&discovery.grid_shape, &discovery.extent_param_names)?,
            element_count,
        );

        Ok(Self {
            device_params,
            host_params,
            device_call_args,
            prelude: Vec::new(),
            host_prelude,
            launch,
            names,
        })
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
}

struct BoundaryDiscovery {
    grid_shape: GridShape,
    extent_param_names: HashMap<usize, String>,
    used_host_names: HashSet<String>,
}

fn discover_boundary(boundary: &[&Variable]) -> Result<BoundaryDiscovery, CudaAbiError> {
    let mut grid_shape = None;
    let mut extent_param_names = HashMap::new();
    let mut used_host_names = HashSet::new();

    // This pass only discovers facts needed before ABI emission. In particular,
    // gpu.global sizes may reference extent leaves that appear anywhere in the
    // boundary, so extent names must be known before globals are lowered.
    for variable in boundary {
        if let Some(shape) = GridShape::from_type(&variable.ty)? {
            if grid_shape.replace(shape).is_some() {
                return Err(CudaAbiError::DuplicateGrid);
            }
        }

        // Extents remain host-side launch inputs. gpu.grid and gpu.global
        // dimensions refer to these leaves, so collect their stable parameter
        // names before emitting either launch config or global-size expressions.
        if let Some(leaf) = extent_leaf(&variable.ty) {
            let name = unique_name(&sanitize_ident(&variable.name), &mut used_host_names);
            extent_param_names.insert(leaf, name);
        }
    }

    Ok(BoundaryDiscovery {
        grid_shape: grid_shape.ok_or(CudaAbiError::MissingGrid)?,
        extent_param_names,
        used_host_names,
    })
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

impl GridShape {
    fn from_type(obj: &Obj) -> Result<Option<Self>, CudaAbiError> {
        let Some(gpu_grid) = gpu_grid(obj)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            grid: dimension_leaves(gpu_grid.grid).ok_or(CudaAbiError::InvalidGridShape)?,
            block: dimension_leaves(gpu_grid.block).ok_or(CudaAbiError::InvalidGridShape)?,
        }))
    }
}

fn resolve_grid_launch(
    shape: &GridShape,
    extent_param_names: &HashMap<usize, String>,
) -> Result<GridLaunch, CudaAbiError> {
    Ok(GridLaunch {
        grid: resolve_dimension_names(&shape.grid, extent_param_names)?,
        block: resolve_dimension_names(&shape.block, extent_param_names)?,
    })
}

fn resolve_dimension_names(
    leaves: &[usize],
    extent_param_names: &HashMap<usize, String>,
) -> Result<Vec<String>, CudaAbiError> {
    leaves
        .iter()
        .map(|leaf| {
            extent_param_names
                .get(leaf)
                .cloned()
                .ok_or(CudaAbiError::MissingGridExtent(*leaf))
        })
        .collect()
}

fn launch_config(grid: GridLaunch, element_count: Option<String>) -> CudaLaunch {
    let grid_expr = cuda_dim3_expr(&grid.grid);
    let block_expr = cuda_dim3_expr(&grid.block);
    CudaLaunch {
        block_expr,
        grid_expr,
        element_count,
    }
}

fn cuda_dim3_expr(dimensions: &[String]) -> String {
    dimensions.join(", ")
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
    // TODO: allow literal dimension values in addition to shared hypergraph
    // leaves. For now each launch dimension must be backed by an extent
    // argument with the same leaf.
    children
        .iter()
        .map(|child| match child {
            Tree::Leaf(leaf, _) => Some(*leaf),
            _ => None,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct GpuGrid<'a> {
    grid: &'a Obj,
    block: &'a Obj,
}

fn gpu_grid(obj: &Obj) -> Result<Option<GpuGrid<'_>>, CudaAbiError> {
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

fn cuda_global_param_type(global: &GpuGlobal<'_>) -> Result<&'static str, CudaAbiError> {
    match global.element {
        "f32" => Ok("float*"),
        _ => Err(CudaAbiError::UnsupportedGlobalElement(
            global.element.to_string(),
        )),
    }
}

#[derive(Debug, Clone)]
struct GpuGlobal<'a> {
    element: &'a str,
    dimensions: Vec<&'a Obj>,
}

fn gpu_global(obj: &Obj) -> Result<Option<GpuGlobal<'_>>, CudaAbiError> {
    let Some(global) = unwrap_val(obj) else {
        return Ok(None);
    };
    let Tree::Node(global, 0, children) = global else {
        return Ok(None);
    };
    let global_name = global.to_string();

    if global_name == "gpu.global" {
        let [Tree::Node(element, 0, _), dimensions] = children.as_slice() else {
            return Err(CudaAbiError::InvalidGlobalShape);
        };
        let dimensions = global_dimensions(dimensions).ok_or(CudaAbiError::InvalidGlobalShape)?;
        return Ok(Some(GpuGlobal {
            element: element.as_str(),
            dimensions,
        }));
    };

    Ok(None)
}

fn global_size_expr(
    global: &GpuGlobal<'_>,
    extent_param_names: &HashMap<usize, String>,
) -> Result<String, CudaAbiError> {
    let dimensions = global
        .dimensions
        .iter()
        .map(|dimension| dimension_expr(dimension, extent_param_names))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(dimensions.join(" * "))
}

fn dimension_expr(
    dimension: &Obj,
    extent_param_names: &HashMap<usize, String>,
) -> Result<String, CudaAbiError> {
    match dimension {
        Tree::Leaf(leaf, _) => extent_param_names
            .get(leaf)
            .cloned()
            .ok_or(CudaAbiError::MissingGlobalExtent(*leaf)),
        Tree::Node(op, 0, children)
            if matches!(op.to_string().as_str(), "nat.mul" | "*") && children.len() == 2 =>
        {
            let lhs = dimension_expr(&children[0], extent_param_names)?;
            let rhs = dimension_expr(&children[1], extent_param_names)?;
            Ok(format!("({lhs} * {rhs})"))
        }
        _ => {
            // TODO: allow literal dimension values. At the moment every global
            // memory dimension must be expressed in terms of extent-backed
            // hypergraph leaves and supported nat operations.
            Err(CudaAbiError::InvalidGlobalShape)
        }
    }
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
