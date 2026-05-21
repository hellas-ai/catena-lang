//! CUDA ABI construction terminology:
//!
//! - The **kernel** is the `__global__` CUDA function generated from one
//!   Catena entry arrow/program.
//! - **Device** names, parameters, and prelude code belong to that kernel and
//!   are visible while executing on the GPU.
//! - The **host launcher** is the CPU-side wrapper that receives runtime inputs,
//!   computes derived sizes, and invokes the kernel with CUDA launch syntax.
//! - The **source object** is the left-hand side of an arrow type (`source ->
//!   target`). For CUDA kernels, we derive launch and memory information from
//!   the source object of the entry arrow.
//! - The **source parameters** are the variables in that source object. They
//!   are not copied directly to CUDA. Instead, each parameter contributes to
//!   the host ABI, device ABI, launch configuration, memory declarations, or
//!   name rewrites.
//! - A **kernel interface** is the structured contract discovered from those
//!   source parameters: grid shape, global/shared memory resources, extents,
//!   and compile-time constants supplied through CUDA options.
//!
//! This module assembles those pieces into `CudaKernelAbi`, which is then used
//! by CUDA rendering and domain lowering.
//!
//! ABI construction follows a small pipeline:
//!
//! - discover the kernel interface from the entry arrow input,
//! - decide which extents must be passed through to device code,
//! - collect host launcher and device kernel ABI pieces from source parameters,
//! - derive CUDA launch and shared-memory configuration,
//! - analyze views after final CUDA names are known.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::{
    compile::{
        cuda::{
            CudaOptions,
            boundary::{KernelInterface, discover_kernel_interface},
            launch::launch_from_grid_contract,
            parameters::SourceParameterContribution,
            resources::SharedIndexing,
            resources::{SharedMemory, SharedMemoryLayout, bind_global, bind_static_shared},
            util::{sanitize_ident, unique_name},
            views::{ViewAnalysis, extents_required_by_device_code},
        },
        program::{Definition, Variable},
    },
    structured::ir::{Param, StructuredProgram},
};

#[derive(Debug, Clone)]
pub(super) struct CudaKernelAbi {
    pub(super) kernel_params: Vec<Param>,
    pub(super) launcher_params: Vec<Param>,
    pub(super) kernel_arguments: Vec<String>,
    pub(super) kernel_prelude: Vec<String>,
    pub(super) launcher_prelude: Vec<String>,
    pub(super) macros: Vec<CudaMacro>,
    pub(super) launch: CudaLaunch,
    pub(super) dynamic_shared_memory_bytes: Option<String>,
    views: ViewAnalysis,
    cuda_names: HashMap<String, String>,
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
    #[error("extent leaf {0} is not named by any CUDA kernel source parameter")]
    MissingExtentName(usize),
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

        // Read the source object of the entry arrow as the CUDA kernel
        // interface. This discovers the launch contract (`gpu.grid`), memory
        // resources (`gpu.global` / `gpu.shared`), and the extent leaves that
        // give names to their dimensions. We do this before processing any
        // single parameter because shapes can reference extent leaves declared
        // elsewhere in the same source object.
        let kernel_interface = discover_kernel_interface(&source_params, options)?;

        // Find extent parameters that must still be available inside the
        // generated kernel body. Most extents are host-only: they size memory
        // or compute launch dimensions. Some primitives, such as
        // `gpu.view.group`, use an extent to compute per-thread indices in
        // device code, so those extents must also become kernel parameters.
        let extents_required_by_device_code = extents_required_by_device_code(program);

        // Turn each source parameter into concrete ABI pieces: host launcher
        // parameters, device kernel parameters, call arguments, generated
        // prelude code, and name rewrites used during rendering.
        let source_parameter_abi = collect_source_parameter_abi(
            &source_params,
            kernel_interface,
            extents_required_by_device_code,
        )?;

        // Build the host launch expression from the `gpu.grid` contract. The
        // grid itself is not passed to the kernel; it defines the
        // `<<<grid, block, shared_bytes>>>` launch configuration.
        let launch = launch_from_grid_contract(
            &source_parameter_abi.kernel_interface.grid_shape,
            &source_parameter_abi.kernel_interface.extent_cuda_names,
            source_parameter_abi.element_count.clone(),
        )?;

        // Collect any dynamic shared-memory byte count requested by
        // `gpu.shared` parameters. Static shared memory has already emitted
        // declarations in the device prelude and does not contribute here.
        let dynamic_shared_memory_bytes = source_parameter_abi.shared_layout.dynamic_shared_bytes();

        // Analyze view/resource relationships after names are known. Static
        // shared arrays need structured coordinates (`view_x`, `view_y`, ...);
        // dynamic shared/global memory continue to use flat linear indices.
        let views = ViewAnalysis::new(
            program,
            &source_parameter_abi.names,
            source_parameter_abi.shared_indexing.clone(),
        );

        Ok(CudaKernelAbi {
            kernel_params: source_parameter_abi.device_params,
            launcher_params: source_parameter_abi.host_params,
            kernel_arguments: source_parameter_abi.device_call_args,
            kernel_prelude: source_parameter_abi.prelude,
            launcher_prelude: source_parameter_abi.host_prelude,
            macros: source_parameter_abi.kernel_interface.macros,
            launch,
            dynamic_shared_memory_bytes,
            views,
            cuda_names: source_parameter_abi.names,
        })
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.cuda_names
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

struct SourceParameterAbi {
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

struct SourceParameterAbiState {
    source_parameter_abi: SourceParameterAbi,
    used_device_names: HashSet<String>,
    extents_required_by_device_code: HashSet<String>,
    emitted_extent_params: HashSet<usize>,
}

fn collect_source_parameter_abi(
    source_params: &[&Variable],
    kernel_interface: KernelInterface,
    extents_required_by_device_code: HashSet<String>,
) -> Result<SourceParameterAbi, CudaAbiError> {
    let mut state = SourceParameterAbiState {
        source_parameter_abi: SourceParameterAbi {
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
        extents_required_by_device_code,
        emitted_extent_params: HashSet::new(),
    };

    // Source parameters contribute pieces to both the host launcher ABI and the
    // device kernel ABI. We process them in source order so generated signatures
    // and call arguments remain stable.
    for source_param in source_params {
        state.record_source_parameter_contribution(source_param)?;
    }

    Ok(state.source_parameter_abi)
}

impl SourceParameterAbiState {
    fn record_source_parameter_contribution(
        &mut self,
        source_param: &Variable,
    ) -> Result<(), CudaAbiError> {
        match SourceParameterContribution::classify(source_param)? {
            SourceParameterContribution::RuntimeOrStaticExtent { leaf } => {
                self.record_extent_contribution(source_param, leaf)
            }
            SourceParameterContribution::GlobalMemory(global) => {
                self.record_global_memory_contribution(source_param, &global)
            }
            SourceParameterContribution::SharedMemory(shared) => {
                self.record_shared_memory_contribution(source_param, &shared)
            }
            SourceParameterContribution::LaunchGrid => {
                // The grid is a launch contract, not a kernel parameter.
                Ok(())
            }
        }
    }

    fn record_extent_contribution(
        &mut self,
        source_param: &Variable,
        leaf: usize,
    ) -> Result<(), CudaAbiError> {
        let Some(host_or_static_name) = self
            .source_parameter_abi
            .kernel_interface
            .extent_cuda_names
            .get(&leaf)
            .cloned()
        else {
            return Err(CudaAbiError::MissingExtentName(leaf));
        };
        self.source_parameter_abi
            .names
            .insert(source_param.name.clone(), host_or_static_name.clone());

        if self
            .source_parameter_abi
            .kernel_interface
            .compile_time_extent_leaves
            .contains(&leaf)
        {
            return Ok(());
        }

        if self.emitted_extent_params.insert(leaf) {
            self.source_parameter_abi.host_params.push(Param {
                ty: "uint64_t".to_string(),
                name: host_or_static_name.clone(),
            });
        }

        // Most extents exist only to compute launch parameters and memory sizes
        // on the host. gpu.view.group also needs its thread-count extent inside
        // the kernel, so pass that one through as a device parameter.
        if self
            .extents_required_by_device_code
            .contains(&source_param.name)
        {
            let device_name = unique_name(
                &sanitize_ident(&source_param.name),
                &mut self.used_device_names,
            );
            self.source_parameter_abi
                .names
                .insert(source_param.name.clone(), device_name.clone());
            self.source_parameter_abi.device_params.push(Param {
                ty: "uint64_t".to_string(),
                name: device_name,
            });
            self.source_parameter_abi
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
            &self.source_parameter_abi.kernel_interface.extent_cuda_names,
            &mut self.used_device_names,
            &mut self
                .source_parameter_abi
                .kernel_interface
                .reserved_host_names,
        )?;

        self.source_parameter_abi
            .names
            .insert(source_param.name.clone(), binding.device_name.clone());
        self.source_parameter_abi
            .element_count
            .get_or_insert_with(|| binding.size_name.clone());
        self.source_parameter_abi
            .device_params
            .extend(binding.device_params);
        self.source_parameter_abi
            .host_params
            .extend(binding.host_params);
        self.source_parameter_abi
            .host_prelude
            .extend(binding.host_prelude);
        self.source_parameter_abi
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
            &self.source_parameter_abi.kernel_interface.extent_cuda_names,
            &self
                .source_parameter_abi
                .kernel_interface
                .compile_time_extent_leaves,
        )?;

        // This branch is the shared-memory rule:
        // - all dimensions static => CUDA `__shared__ T name[d0][d1]...`
        // - otherwise => a flat slice of dynamic `extern __shared__`
        let binding = match memory {
            SharedMemory::Static(memory) => bind_static_shared(device_name, memory),
            SharedMemory::Dynamic(memory) => self.source_parameter_abi.shared_layout.bind_dynamic(
                device_name,
                memory,
                &mut self.used_device_names,
                &mut self
                    .source_parameter_abi
                    .kernel_interface
                    .reserved_host_names,
            ),
        };

        self.source_parameter_abi
            .names
            .insert(source_param.name.clone(), binding.device_name.clone());
        self.source_parameter_abi
            .shared_indexing
            .insert(binding.device_name.clone(), binding.indexing);
        self.source_parameter_abi
            .device_params
            .extend(binding.device_params);
        self.source_parameter_abi
            .host_prelude
            .extend(binding.host_prelude);
        self.source_parameter_abi
            .device_call_args
            .extend(binding.device_call_args);
        self.source_parameter_abi
            .prelude
            .extend(binding.device_prelude);
        Ok(())
    }
}
