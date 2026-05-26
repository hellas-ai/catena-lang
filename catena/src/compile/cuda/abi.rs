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
        proof::ProofEvidence,
    },
    structured::ir::{Param, Primitive, Stmt, StructuredProgram},
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
    access_certificates: HashMap<(String, String), String>,
    view_ranks: HashMap<String, usize>,
    shape_values: HashMap<String, Vec<String>>,
    grid_views: HashSet<String>,
    global_shapes: HashMap<String, Vec<String>>,
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
        proof_evidence: Option<&ProofEvidence>,
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
        // or compute launch dimensions. View layout primitives can also need
        // extents to compute device-side coordinates.
        let mut extents_required_by_device_code = extents_required_by_device_code(program);
        if program_uses_tiled_view(program) || program_uses_grid_view_global_access(program) {
            for source_param in &source_params {
                if crate::compile::cuda::shape::extent_leaf(&source_param.ty).is_some() {
                    extents_required_by_device_code.insert(source_param.name.clone());
                }
            }
        }

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
        let mut global_shapes = source_parameter_abi.global_shapes.clone();
        let view_metadata =
            collect_view_metadata(program, &source_parameter_abi.names, &mut global_shapes);
        let access_certificates = collect_access_certificates(
            &definition.name,
            program,
            &source_parameter_abi.names,
            proof_evidence,
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
            access_certificates,
            view_ranks: view_metadata.ranks,
            shape_values: view_metadata.shapes,
            grid_views: view_metadata.grid_views,
            global_shapes,
        })
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.cuda_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    pub(super) fn shared_access(&self, shared: &str, view: &str) -> String {
        if self.is_grid_view(view) && self.static_view_rank(view).is_none() {
            return format!("{shared}[{view}_thread.x]");
        }
        self.views.shared_access(shared, view)
    }

    pub(super) fn static_view_rank(&self, view: &str) -> Option<usize> {
        self.views.static_view_rank(view)
    }

    pub(super) fn access_certificate(&self, memory: &str, view: &str) -> Option<&str> {
        self.access_certificates
            .get(&(memory.to_string(), view.to_string()))
            .map(String::as_str)
    }

    pub(super) fn is_grid_view(&self, view: &str) -> bool {
        self.grid_views.contains(view)
    }

    pub(super) fn shape_value(&self, shape: &str) -> Option<&[String]> {
        self.shape_values.get(shape).map(Vec::as_slice)
    }

    pub(super) fn global_access(&self, global: &str, view: &str) -> String {
        match (self.view_ranks.get(view), self.global_shapes.get(global)) {
            (_, Some(shape)) if self.is_grid_view(view) && shape.len() == 2 => {
                format!("{global}[{view}_block.x * {} + {view}_thread.x]", shape[1])
            }
            (Some(2), Some(shape)) if shape.len() == 2 => {
                format!("{global}[{view}_row * {} + {view}_col]", shape[1])
            }
            _ => format!("{global}[{view}]"),
        }
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

fn program_uses_tiled_view(program: &StructuredProgram) -> bool {
    stmts_use_tiled_view(&program.body)
}

fn stmts_use_tiled_view(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
            stmts_use_tiled_view(body)
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => stmts_use_tiled_view(then_body) || stmts_use_tiled_view(else_body),
        Stmt::Switch { cases, .. } => cases.iter().any(|case| stmts_use_tiled_view(case)),
        Stmt::Primitive(primitive) => primitive.name == "gpu.view.group-by-tile",
        Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::Return
        | Stmt::Barrier
        | Stmt::Assign { .. }
        | Stmt::Comment(_) => false,
    })
}

fn program_uses_grid_view_global_access(program: &StructuredProgram) -> bool {
    let mut grid_views = HashSet::new();
    stmts_collect_grid_views(&program.body, &mut grid_views);
    stmts_use_grid_view_global_access(&program.body, &grid_views)
}

fn stmts_collect_grid_views(stmts: &[Stmt], grid_views: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                stmts_collect_grid_views(body, grid_views);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                stmts_collect_grid_views(then_body, grid_views);
                stmts_collect_grid_views(else_body, grid_views);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    stmts_collect_grid_views(case, grid_views);
                }
            }
            Stmt::Primitive(primitive) if primitive.name == "gpu.grid.view" => {
                if let Some(view) = primitive.outputs.first() {
                    grid_views.insert(view.clone());
                }
            }
            Stmt::Primitive(_)
            | Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn stmts_use_grid_view_global_access(stmts: &[Stmt], grid_views: &HashSet<String>) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
            stmts_use_grid_view_global_access(body, grid_views)
        }
        Stmt::If {
            then_body,
            else_body,
            ..
        } => {
            stmts_use_grid_view_global_access(then_body, grid_views)
                || stmts_use_grid_view_global_access(else_body, grid_views)
        }
        Stmt::Switch { cases, .. } => cases
            .iter()
            .any(|case| stmts_use_grid_view_global_access(case, grid_views)),
        Stmt::Primitive(primitive)
            if primitive.name == "gpu.global.load" || primitive.name == "gpu.global.store" =>
        {
            primitive
                .inputs
                .get(1)
                .is_some_and(|view| grid_views.contains(view))
        }
        Stmt::Primitive(_)
        | Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::Return
        | Stmt::Barrier
        | Stmt::Assign { .. }
        | Stmt::Comment(_) => false,
    })
}

fn collect_view_metadata(
    program: &StructuredProgram,
    names: &HashMap<String, String>,
    global_shapes: &mut HashMap<String, Vec<String>>,
) -> CudaViewMetadata {
    let mut grid_views = HashSet::new();
    let mut view_ranks = HashMap::new();
    let mut shape_values = HashMap::new();
    collect_view_metadata_inputs(
        &program.body,
        names,
        global_shapes,
        &mut grid_views,
        &mut view_ranks,
        &mut shape_values,
    );

    CudaViewMetadata {
        ranks: view_ranks,
        shapes: shape_values,
        grid_views,
    }
}

struct CudaViewMetadata {
    ranks: HashMap<String, usize>,
    shapes: HashMap<String, Vec<String>>,
    grid_views: HashSet<String>,
}

fn collect_view_metadata_inputs(
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    global_shapes: &mut HashMap<String, Vec<String>>,
    grid_views: &mut HashSet<String>,
    view_ranks: &mut HashMap<String, usize>,
    shape_values: &mut HashMap<String, Vec<String>>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_view_metadata_inputs(
                    body,
                    names,
                    global_shapes,
                    grid_views,
                    view_ranks,
                    shape_values,
                );
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_view_metadata_inputs(
                    then_body,
                    names,
                    global_shapes,
                    grid_views,
                    view_ranks,
                    shape_values,
                );
                collect_view_metadata_inputs(
                    else_body,
                    names,
                    global_shapes,
                    grid_views,
                    view_ranks,
                    shape_values,
                );
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_view_metadata_inputs(
                        case,
                        names,
                        global_shapes,
                        grid_views,
                        view_ranks,
                        shape_values,
                    );
                }
            }
            Stmt::Primitive(primitive) => {
                collect_primitive_view_metadata(
                    primitive,
                    names,
                    global_shapes,
                    grid_views,
                    view_ranks,
                    shape_values,
                );
            }
            Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_primitive_view_metadata(
    primitive: &Primitive,
    names: &HashMap<String, String>,
    global_shapes: &mut HashMap<String, Vec<String>>,
    grid_views: &mut HashSet<String>,
    view_ranks: &mut HashMap<String, usize>,
    shape_values: &mut HashMap<String, Vec<String>>,
) {
    if primitive.name == "gpu.grid.view" {
        if let Some(view) = primitive.outputs.first() {
            let view = rename_with(names, view);
            grid_views.insert(view);
        }
        return;
    }

    if primitive.name == "gpu.shape.row" {
        if let (Some(cols), Some(shape)) = (primitive.inputs.first(), primitive.outputs.first()) {
            shape_values.insert(
                rename_with(names, shape),
                vec!["1".to_string(), rename_with(names, cols)],
            );
        }
        return;
    }

    if primitive.name == "gpu.shape.row-mul" {
        if let ([lhs, rhs], Some(shape)) = (primitive.inputs.as_slice(), primitive.outputs.first())
        {
            shape_values.insert(
                rename_with(names, shape),
                vec![
                    "1".to_string(),
                    format!(
                        "({} * {})",
                        rename_with(names, lhs),
                        rename_with(names, rhs)
                    ),
                ],
            );
        }
        return;
    }

    if primitive.name == "gpu.shape.col" {
        if let (Some(rows), Some(shape)) = (primitive.inputs.first(), primitive.outputs.first()) {
            shape_values.insert(
                rename_with(names, shape),
                vec![rename_with(names, rows), "1".to_string()],
            );
        }
        return;
    }

    if primitive.name == "gpu.shape.col-mul" {
        if let ([lhs, rhs], Some(shape)) = (primitive.inputs.as_slice(), primitive.outputs.first())
        {
            shape_values.insert(
                rename_with(names, shape),
                vec![
                    format!(
                        "({} * {})",
                        rename_with(names, lhs),
                        rename_with(names, rhs)
                    ),
                    "1".to_string(),
                ],
            );
        }
        return;
    }

    if primitive.name == "gpu.shape.2d" {
        if let ([rows, cols], Some(shape)) =
            (primitive.inputs.as_slice(), primitive.outputs.first())
        {
            shape_values.insert(
                rename_with(names, shape),
                vec![rename_with(names, rows), rename_with(names, cols)],
            );
        }
        return;
    }

    if primitive.name == "gpu.view.group-by-tile" || primitive.name == "gpu.view.group" {
        if let Some(view) = primitive.outputs.first() {
            let view = rename_with(names, view);
            view_ranks.insert(view, 2);
        }
        return;
    }

    if primitive.name == "gpu.view.reshape" {
        if let Some(view) = primitive.outputs.first() {
            let view = rename_with(names, view);
            view_ranks.insert(view, 2);
        }
        return;
    }

    if primitive.name == "gpu.view.row" || primitive.name == "gpu.view.col" {
        if let Some(view) = primitive.outputs.first() {
            let view = rename_with(names, view);
            view_ranks.insert(view, 1);
        }
        return;
    }

    if primitive.name == "gpu.global.store" {
        let Some(global) = primitive.inputs.first() else {
            return;
        };
        let global = rename_with(names, global);

        if let Some(shape) = global_shapes.get(&global).cloned() {
            if let Some(output) = primitive.outputs.first() {
                let output = rename_with(names, output);
                global_shapes.insert(output, shape);
            }
        }
    }
}

fn collect_access_certificates(
    entry: &str,
    program: &StructuredProgram,
    names: &HashMap<String, String>,
    proof_evidence: Option<&ProofEvidence>,
) -> HashMap<(String, String), String> {
    let mut certificates = HashMap::new();
    let Some(proof_evidence) = proof_evidence else {
        return certificates;
    };

    collect_access_certificates_from_stmts(
        entry,
        &program.body,
        names,
        proof_evidence,
        &mut certificates,
    );
    certificates
}

fn collect_access_certificates_from_stmts(
    entry: &str,
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    proof_evidence: &ProofEvidence,
    certificates: &mut HashMap<(String, String), String>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_access_certificates_from_stmts(
                    entry,
                    body,
                    names,
                    proof_evidence,
                    certificates,
                );
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_access_certificates_from_stmts(
                    entry,
                    then_body,
                    names,
                    proof_evidence,
                    certificates,
                );
                collect_access_certificates_from_stmts(
                    entry,
                    else_body,
                    names,
                    proof_evidence,
                    certificates,
                );
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_access_certificates_from_stmts(
                        entry,
                        case,
                        names,
                        proof_evidence,
                        certificates,
                    );
                }
            }
            Stmt::Primitive(primitive) => {
                collect_primitive_access_certificate(
                    entry,
                    primitive,
                    names,
                    proof_evidence,
                    certificates,
                );
            }
            Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_primitive_access_certificate(
    entry: &str,
    primitive: &Primitive,
    names: &HashMap<String, String>,
    proof_evidence: &ProofEvidence,
    certificates: &mut HashMap<(String, String), String>,
) {
    if !is_gpu_memory_access(&primitive.name) {
        return;
    }
    let Some(memory) = primitive.inputs.first() else {
        return;
    };
    let Some(view) = primitive.inputs.get(1) else {
        return;
    };

    let proof_name = format!("{entry}.{memory}.{view}.safe-access");
    if proof_evidence.has_definition(&proof_name) {
        certificates.insert(
            (rename_with(names, memory), rename_with(names, view)),
            proof_name,
        );
    }
}

fn is_gpu_memory_access(name: &str) -> bool {
    matches!(
        name,
        "gpu.global.load" | "gpu.global.store" | "gpu.shared.load" | "gpu.shared.store"
    )
}

fn rename_with(names: &HashMap<String, String>, name: &str) -> String {
    names.get(name).cloned().unwrap_or_else(|| name.to_string())
}

struct SourceParameterAbi {
    kernel_interface: KernelInterface,
    device_params: Vec<Param>,
    host_params: Vec<Param>,
    device_call_args: Vec<String>,
    prelude: Vec<String>,
    host_prelude: Vec<String>,
    names: HashMap<String, String>,
    global_shapes: HashMap<String, Vec<String>>,
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
            global_shapes: HashMap::new(),
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
        // on the host. View layout primitives can also need extents inside the
        // kernel, so pass those through as device parameters.
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
            .global_shapes
            .insert(binding.device_name.clone(), binding.dimensions.clone());
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
