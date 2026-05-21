use std::collections::{HashMap, HashSet};

use crate::{
    compile::{
        cuda::{
            abi::CudaAbiError,
            boundary::{GpuGlobal, GpuShared},
            shape::{ShapeExpr, shape_expr},
            util::{sanitize_ident, unique_name},
        },
        program::Variable,
    },
    structured::ir::Param,
};

#[derive(Debug, Clone)]
pub(super) enum SharedIndexing {
    Flat,
    Static { rank: usize },
}

pub(super) struct GlobalBinding {
    pub(super) device_name: String,
    pub(super) size_name: String,
    pub(super) dimensions: Vec<String>,
    pub(super) device_params: Vec<Param>,
    pub(super) host_params: Vec<Param>,
    pub(super) host_prelude: Vec<String>,
    pub(super) device_call_args: Vec<String>,
}

pub(super) struct SharedBinding {
    pub(super) device_name: String,
    pub(super) indexing: SharedIndexing,
    pub(super) device_params: Vec<Param>,
    pub(super) host_prelude: Vec<String>,
    pub(super) device_call_args: Vec<String>,
    pub(super) device_prelude: Vec<String>,
}

pub(super) fn bind_global(
    variable: &Variable,
    global: &GpuGlobal<'_>,
    extent_names: &HashMap<usize, String>,
    used_device_names: &mut HashSet<String>,
    used_host_names: &mut HashSet<String>,
) -> Result<GlobalBinding, CudaAbiError> {
    let param_ty = cuda_global_param_type(global)?;
    let base_name = sanitize_ident(&variable.name);
    let device_name = unique_name(&base_name, used_device_names);
    let host_name = unique_name(&base_name, used_host_names);
    let size_name = unique_name(&format!("{device_name}_size"), used_device_names);
    let size_expr = memory_size_expr(
        &global.dimensions,
        extent_names,
        CudaAbiError::MissingGlobalExtent,
        || CudaAbiError::InvalidGlobalShape,
    )?;
    let dimensions = memory_dimension_exprs(
        &global.dimensions,
        extent_names,
        CudaAbiError::MissingGlobalExtent,
        || CudaAbiError::InvalidGlobalShape,
    )?;

    Ok(GlobalBinding {
        device_name: device_name.clone(),
        size_name: size_name.clone(),
        dimensions,
        device_params: vec![
            Param {
                ty: "uint64_t".to_string(),
                name: size_name.clone(),
            },
            Param {
                ty: param_ty.to_string(),
                name: device_name,
            },
        ],
        host_params: vec![Param {
            ty: param_ty.to_string(),
            name: host_name.clone(),
        }],
        host_prelude: vec![format!("uint64_t {size_name} = {size_expr};")],
        device_call_args: vec![size_name, host_name],
    })
}

pub(super) enum SharedMemory {
    Static(StaticSharedMemory),
    Dynamic(DynamicSharedMemory),
}

pub(super) struct StaticSharedMemory {
    element_ty: &'static str,
    shape: ShapeExpr,
}

pub(super) struct DynamicSharedMemory {
    element_ty: &'static str,
    size_expr: String,
}

impl SharedMemory {
    pub(super) fn from_gpu_shared(
        shared: &GpuShared<'_>,
        extent_names: &HashMap<usize, String>,
        static_extent_leaves: &HashSet<usize>,
    ) -> Result<Self, CudaAbiError> {
        let element_ty = cuda_shared_element_type(shared)?;
        let shape = shape_expr(
            &shared.dimensions,
            extent_names,
            static_extent_leaves,
            CudaAbiError::MissingSharedExtent,
            || CudaAbiError::InvalidSharedShape,
        )?;

        if shape.is_static() {
            Ok(Self::Static(StaticSharedMemory { element_ty, shape }))
        } else {
            Ok(Self::Dynamic(DynamicSharedMemory {
                element_ty,
                size_expr: shape.product_expr(),
            }))
        }
    }
}

// Tracks only the dynamic shared-memory region. Static shared allocations are
// independent CUDA `__shared__` arrays, while dynamic allocations are adjacent
// slices of CUDA's single `extern __shared__` buffer.
pub(super) struct SharedMemoryLayout {
    device_size_names: Vec<String>,
    host_size_names: Vec<String>,
}

impl SharedMemoryLayout {
    pub(super) fn new() -> Self {
        Self {
            device_size_names: Vec::new(),
            host_size_names: Vec::new(),
        }
    }

    pub(super) fn bind_dynamic(
        &mut self,
        device_name: String,
        memory: DynamicSharedMemory,
        used_device_names: &mut HashSet<String>,
        used_host_names: &mut HashSet<String>,
    ) -> SharedBinding {
        let device_size_name = unique_name(&format!("{device_name}_size"), used_device_names);
        let host_size_name = unique_name(&format!("{device_name}_size"), used_host_names);
        let offset_expr = self.device_offset_expr();

        let mut device_prelude = Vec::new();
        if self.device_size_names.is_empty() {
            device_prelude.push(format!(
                "extern __shared__ {} __shared_mem[];",
                memory.element_ty
            ));
            device_prelude.push(format!(
                "{}* {device_name} = __shared_mem;",
                memory.element_ty
            ));
        } else {
            device_prelude.push(format!(
                "{}* {device_name} = __shared_mem + ({offset_expr});",
                memory.element_ty
            ));
        }

        self.device_size_names.push(device_size_name.clone());
        self.host_size_names.push(host_size_name.clone());

        SharedBinding {
            device_name,
            indexing: SharedIndexing::Flat,
            device_params: vec![Param {
                ty: "uint64_t".to_string(),
                name: device_size_name,
            }],
            host_prelude: vec![format!("uint64_t {host_size_name} = {};", memory.size_expr)],
            device_call_args: vec![host_size_name],
            device_prelude,
        }
    }

    pub(super) fn dynamic_shared_bytes(&self) -> Option<String> {
        if self.host_size_names.is_empty() {
            return None;
        }
        Some(format!(
            "({}) * sizeof(float)",
            self.host_size_names.join(" + ")
        ))
    }

    fn device_offset_expr(&self) -> String {
        self.device_size_names.join(" + ")
    }
}

pub(super) fn bind_static_shared(device_name: String, memory: StaticSharedMemory) -> SharedBinding {
    SharedBinding {
        device_name: device_name.clone(),
        indexing: SharedIndexing::Static {
            rank: memory.shape.rank(),
        },
        device_params: Vec::new(),
        host_prelude: Vec::new(),
        device_call_args: Vec::new(),
        device_prelude: vec![format!(
            "__shared__ {} {device_name}{};",
            memory.element_ty,
            memory.shape.cuda_array_suffix()
        )],
    }
}

fn memory_size_expr(
    dimensions: &[&crate::lang::Obj],
    extent_names: &HashMap<usize, String>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<String, CudaAbiError> {
    let dimensions = dimensions
        .iter()
        .map(|dimension| {
            crate::compile::cuda::shape::dimension_expr(
                dimension,
                extent_names,
                missing_extent,
                invalid_shape,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(dimensions.join(" * "))
}

fn memory_dimension_exprs(
    dimensions: &[&crate::lang::Obj],
    extent_names: &HashMap<usize, String>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<Vec<String>, CudaAbiError> {
    dimensions
        .iter()
        .map(|dimension| {
            crate::compile::cuda::shape::dimension_expr(
                dimension,
                extent_names,
                missing_extent,
                invalid_shape,
            )
        })
        .collect()
}

fn cuda_global_param_type(global: &GpuGlobal<'_>) -> Result<&'static str, CudaAbiError> {
    match global.element {
        "f32" => Ok("float*"),
        _ => Err(CudaAbiError::UnsupportedGlobalElement(
            global.element.to_string(),
        )),
    }
}

fn cuda_shared_element_type(shared: &GpuShared<'_>) -> Result<&'static str, CudaAbiError> {
    match shared.element {
        "f32" => Ok("float"),
        _ => Err(CudaAbiError::UnsupportedSharedElement(
            shared.element.to_string(),
        )),
    }
}
