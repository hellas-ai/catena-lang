use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::{
    compile::{
        cuda::{
            CudaOptions,
            boundary::{
                GridShape, KernelInterface, discover_kernel_interface, gpu_global, gpu_grid,
                gpu_shared,
            },
            resources::SharedIndexing,
            resources::{SharedMemory, SharedMemoryLayout, bind_global, bind_static_shared},
            shape::{dimension_expr, extent_leaf},
            util::{sanitize_ident, unique_name},
            views::{ViewAnalysis, device_extent_names},
        },
        program::{Definition, Variable},
    },
    structured::ir::{Param, StructuredProgram},
};

#[derive(Debug, Clone)]
pub(super) struct CudaKernelAbi {
    pub(super) device_params: Vec<Param>,
    pub(super) host_params: Vec<Param>,
    pub(super) device_call_args: Vec<String>,
    pub(super) prelude: Vec<String>,
    pub(super) host_prelude: Vec<String>,
    pub(super) macros: Vec<CudaMacro>,
    pub(super) launch: CudaLaunch,
    pub(super) dynamic_shared_bytes: Option<String>,
    views: ViewAnalysis,
    names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct CudaMacro {
    pub(super) name: String,
    pub(super) value: u64,
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
    #[error("CUDA kernel source parameters are missing a gpu.grid value")]
    MissingGrid,
    #[error("CUDA kernel source parameters must provide exactly one gpu.grid value")]
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
    #[error(
        "gpu.shared boundary values must be gpu.shared element dimensions with 1d, 2d, or 3d dimensions"
    )]
    InvalidSharedShape,
    #[error("gpu.shared dimension leaf {0} is not backed by an extent argument")]
    MissingSharedExtent(usize),
    #[error("unsupported CUDA shared memory element type `{0}`")]
    UnsupportedSharedElement(String),
    #[error("unsupported CUDA global memory element type `{0}`")]
    UnsupportedGlobalElement(String),
    #[error("--cuda-static `{0}` does not match any CUDA kernel source parameter")]
    UnknownStaticValue(String),
    #[error("--cuda-static `{0}` was provided for a non-extent CUDA source parameter")]
    StaticValueNotExtent(String),
    #[error("unsupported CUDA kernel source parameter `{name}` of type `{ty}`")]
    UnsupportedSourceParameter { name: String, ty: String },
}

impl CudaKernelAbi {
    pub(super) fn from_definition(
        definition: &Definition,
        program: &StructuredProgram,
        options: &CudaOptions,
    ) -> Result<Self, CudaAbiError> {
        let source_params = source_parameters(definition)?;
        // The definition params are the source variables of the entry arrow.
        // We read them as the kernel interface before processing any one param:
        // grid/global/shared shapes may use extent leaves declared elsewhere in
        // the source object.
        let kernel_interface = discover_kernel_interface(&source_params, options)?;
        let device_extent_names = device_extent_names(program);
        let source_parameter_contributions = collect_source_parameter_contributions(
            &source_params,
            kernel_interface,
            device_extent_names,
        )?;
        let launch = launch_from_grid_contract(
            &source_parameter_contributions.kernel_interface.grid_shape,
            &source_parameter_contributions.kernel_interface.extent_names,
            source_parameter_contributions.element_count.clone(),
        )?;
        let dynamic_shared_bytes = source_parameter_contributions
            .shared_layout
            .dynamic_shared_bytes();
        let views = ViewAnalysis::new(
            program,
            &source_parameter_contributions.names,
            source_parameter_contributions.shared_indexing.clone(),
        );

        Ok(CudaKernelAbi {
            device_params: source_parameter_contributions.device_params,
            host_params: source_parameter_contributions.host_params,
            device_call_args: source_parameter_contributions.device_call_args,
            prelude: source_parameter_contributions.prelude,
            host_prelude: source_parameter_contributions.host_prelude,
            macros: source_parameter_contributions.kernel_interface.macros,
            launch,
            dynamic_shared_bytes,
            views,
            names: source_parameter_contributions.names,
        })
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    pub(super) fn shared_access(&self, shared: &str, view: &str) -> String {
        self.views.shared_access(shared, view)
    }

    pub(super) fn static_view_rank(&self, view: &str) -> Option<usize> {
        self.views.static_view_rank(view)
    }
}

fn source_parameters(definition: &Definition) -> Result<Vec<&Variable>, CudaAbiError> {
    definition
        .params
        .iter()
        .map(|id| {
            definition
                .context
                .variable(*id)
                .ok_or(CudaAbiError::MissingParamVariable(*id))
        })
        .collect()
}

struct SourceParameterContributions {
    kernel_interface: KernelInterface,
    device_params: Vec<Param>,
    host_params: Vec<Param>,
    device_call_args: Vec<String>,
    prelude: Vec<String>,
    host_prelude: Vec<String>,
    names: HashMap<String, String>,
    element_count: Option<String>,
    shared_layout: SharedMemoryLayout,
    shared_indexing: HashMap<String, SharedIndexing>,
}

struct SourceParameterContributionState {
    contributions: SourceParameterContributions,
    used_device_names: HashSet<String>,
    device_extent_names: HashSet<String>,
    emitted_extent_params: HashSet<usize>,
}

fn collect_source_parameter_contributions(
    source_params: &[&Variable],
    kernel_interface: KernelInterface,
    device_extent_names: HashSet<String>,
) -> Result<SourceParameterContributions, CudaAbiError> {
    let mut state = SourceParameterContributionState {
        contributions: SourceParameterContributions {
            kernel_interface,
            device_params: Vec::new(),
            host_params: Vec::new(),
            device_call_args: Vec::new(),
            prelude: Vec::new(),
            host_prelude: Vec::new(),
            names: HashMap::new(),
            element_count: None,
            shared_layout: SharedMemoryLayout::new(),
            shared_indexing: HashMap::new(),
        },
        used_device_names: HashSet::new(),
        device_extent_names,
        emitted_extent_params: HashSet::new(),
    };

    // Source parameters contribute pieces to both the host launcher ABI and the
    // device kernel ABI. We process them in source order so generated signatures
    // and call arguments remain stable.
    for source_param in source_params {
        state.record_source_parameter_contribution(source_param)?;
    }

    Ok(state.contributions)
}

impl SourceParameterContributionState {
    fn record_source_parameter_contribution(
        &mut self,
        source_param: &Variable,
    ) -> Result<(), CudaAbiError> {
        if let Some(leaf) = extent_leaf(&source_param.ty) {
            self.record_extent_contribution(source_param, leaf)?;
            return Ok(());
        }
        if let Some(global) = gpu_global(&source_param.ty)? {
            self.record_global_memory_contribution(source_param, &global)?;
            return Ok(());
        }
        if let Some(shared) = gpu_shared(&source_param.ty)? {
            self.record_shared_memory_contribution(source_param, &shared)?;
            return Ok(());
        }
        if gpu_grid(&source_param.ty)?.is_some() {
            // The grid is a launch contract, not a kernel parameter.
            return Ok(());
        }

        Err(CudaAbiError::UnsupportedSourceParameter {
            name: source_param.name.clone(),
            ty: format!("{:?}", source_param.ty),
        })
    }

    fn record_extent_contribution(
        &mut self,
        source_param: &Variable,
        leaf: usize,
    ) -> Result<(), CudaAbiError> {
        let Some(host_or_static_name) = self
            .contributions
            .kernel_interface
            .extent_names
            .get(&leaf)
            .cloned()
        else {
            return Err(CudaAbiError::MissingGridExtent(leaf));
        };
        self.contributions
            .names
            .insert(source_param.name.clone(), host_or_static_name.clone());

        if self
            .contributions
            .kernel_interface
            .static_extent_leaves
            .contains(&leaf)
        {
            return Ok(());
        }

        if self.emitted_extent_params.insert(leaf) {
            self.contributions.host_params.push(Param {
                ty: "uint64_t".to_string(),
                name: host_or_static_name.clone(),
            });
        }

        // Most extents exist only to compute launch parameters and memory sizes
        // on the host. gpu.view.group also needs its thread-count extent inside
        // the kernel, so pass that one through as a device parameter.
        if self.device_extent_names.contains(&source_param.name) {
            let device_name = unique_name(
                &sanitize_ident(&source_param.name),
                &mut self.used_device_names,
            );
            self.contributions
                .names
                .insert(source_param.name.clone(), device_name.clone());
            self.contributions.device_params.push(Param {
                ty: "uint64_t".to_string(),
                name: device_name,
            });
            self.contributions
                .device_call_args
                .push(host_or_static_name);
        }

        Ok(())
    }

    fn record_global_memory_contribution(
        &mut self,
        source_param: &Variable,
        global: &crate::compile::cuda::boundary::GpuGlobal<'_>,
    ) -> Result<(), CudaAbiError> {
        let binding = bind_global(
            source_param,
            global,
            &self.contributions.kernel_interface.extent_names,
            &mut self.used_device_names,
            &mut self.contributions.kernel_interface.used_host_names,
        )?;

        self.contributions
            .names
            .insert(source_param.name.clone(), binding.device_name.clone());
        self.contributions
            .element_count
            .get_or_insert_with(|| binding.size_name.clone());
        self.contributions
            .device_params
            .extend(binding.device_params);
        self.contributions.host_params.extend(binding.host_params);
        self.contributions.host_prelude.extend(binding.host_prelude);
        self.contributions
            .device_call_args
            .extend(binding.device_call_args);
        Ok(())
    }

    fn record_shared_memory_contribution(
        &mut self,
        source_param: &Variable,
        shared: &crate::compile::cuda::boundary::GpuShared<'_>,
    ) -> Result<(), CudaAbiError> {
        let device_name = unique_name(
            &sanitize_ident(&source_param.name),
            &mut self.used_device_names,
        );
        let memory = SharedMemory::from_gpu_shared(
            shared,
            &self.contributions.kernel_interface.extent_names,
            &self.contributions.kernel_interface.static_extent_leaves,
        )?;

        // This branch is the shared-memory rule:
        // - all dimensions static => CUDA `__shared__ T name[d0][d1]...`
        // - otherwise => a flat slice of dynamic `extern __shared__`
        let binding = match memory {
            SharedMemory::Static(memory) => bind_static_shared(device_name, memory),
            SharedMemory::Dynamic(memory) => self.contributions.shared_layout.bind_dynamic(
                device_name,
                memory,
                &mut self.used_device_names,
                &mut self.contributions.kernel_interface.used_host_names,
            ),
        };

        self.contributions
            .names
            .insert(source_param.name.clone(), binding.device_name.clone());
        self.contributions
            .shared_indexing
            .insert(binding.device_name.clone(), binding.indexing);
        self.contributions
            .device_params
            .extend(binding.device_params);
        self.contributions.host_prelude.extend(binding.host_prelude);
        self.contributions
            .device_call_args
            .extend(binding.device_call_args);
        self.contributions.prelude.extend(binding.device_prelude);
        Ok(())
    }
}

struct GridLaunch {
    grid: Vec<String>,
    block: Vec<String>,
}

fn resolve_grid_launch(
    shape: &GridShape,
    extent_names: &HashMap<usize, String>,
) -> Result<GridLaunch, CudaAbiError> {
    Ok(GridLaunch {
        grid: resolve_dimension_names(&shape.grid, extent_names)?,
        block: resolve_dimension_names(&shape.block, extent_names)?,
    })
}

fn resolve_dimension_names(
    dimensions: &[crate::lang::Obj],
    extent_names: &HashMap<usize, String>,
) -> Result<Vec<String>, CudaAbiError> {
    dimensions
        .iter()
        .map(|dimension| {
            dimension_expr(
                dimension,
                extent_names,
                CudaAbiError::MissingGridExtent,
                || CudaAbiError::InvalidGridShape,
            )
        })
        .collect()
}

fn launch_from_grid_contract(
    grid_shape: &GridShape,
    extent_names: &HashMap<usize, String>,
    element_count: Option<String>,
) -> Result<CudaLaunch, CudaAbiError> {
    let grid = resolve_grid_launch(grid_shape, extent_names)?;
    Ok(cuda_launch_config(grid, element_count))
}

fn cuda_launch_config(grid: GridLaunch, element_count: Option<String>) -> CudaLaunch {
    CudaLaunch {
        block_expr: grid.block.join(", "),
        grid_expr: grid.grid.join(", "),
        element_count,
    }
}
