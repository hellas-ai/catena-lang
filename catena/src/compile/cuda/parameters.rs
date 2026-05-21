use crate::compile::{
    cuda::{
        abi::CudaAbiError,
        boundary::{GpuGlobal, GpuShared, gpu_global, gpu_grid, gpu_shared},
        shape::extent_leaf,
    },
    program::Variable,
};

pub(super) enum SourceParameterContribution<'a> {
    RuntimeOrStaticExtent { leaf: usize },
    LaunchGrid,
    GlobalMemory(GpuGlobal<'a>),
    SharedMemory(GpuShared<'a>),
}

impl<'a> SourceParameterContribution<'a> {
    pub(super) fn classify(source_param: &'a Variable) -> Result<Self, CudaAbiError> {
        if let Some(leaf) = extent_leaf(&source_param.ty) {
            return Ok(Self::RuntimeOrStaticExtent { leaf });
        }
        if let Some(global) = gpu_global(&source_param.ty)? {
            return Ok(Self::GlobalMemory(global));
        }
        if let Some(shared) = gpu_shared(&source_param.ty)? {
            return Ok(Self::SharedMemory(shared));
        }
        if gpu_grid(&source_param.ty)?.is_some() {
            return Ok(Self::LaunchGrid);
        }

        Err(CudaAbiError::UnsupportedSourceParameter {
            name: source_param.name.clone(),
            ty: format!("{:?}", source_param.ty),
        })
    }
}
