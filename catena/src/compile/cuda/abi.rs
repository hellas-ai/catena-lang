use std::collections::{HashMap, HashSet};

use metacat::tree::Tree;
use thiserror::Error;

use crate::{
    compile::{
        cuda::CudaOptions,
        program::{Definition, Variable},
    },
    lang::Obj,
    structured::ir::{Param, Primitive, Stmt, StructuredProgram},
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
    shared_indexing: HashMap<String, SharedIndexing>,
    static_view_ranks: HashMap<String, usize>,
    names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct CudaMacro {
    pub(super) name: String,
    pub(super) value: u64,
}

#[derive(Debug, Clone)]
pub(super) enum SharedIndexing {
    Flat,
    Static { rank: usize },
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
    #[error("--cuda-static `{0}` does not match any CUDA kernel boundary argument")]
    UnknownStaticValue(String),
    #[error("--cuda-static `{0}` was provided for a non-extent CUDA boundary argument")]
    StaticValueNotExtent(String),
    #[error("unsupported CUDA kernel boundary argument `{name}` of type `{ty}`")]
    UnsupportedBoundaryArgument { name: String, ty: String },
}

impl CudaKernelAbi {
    pub(super) fn from_definition(
        definition: &Definition,
        program: &StructuredProgram,
        options: &CudaOptions,
    ) -> Result<Self, CudaAbiError> {
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

        let discovery = discover_boundary(&boundary, options)?;
        let mut used_device_names = HashSet::new();
        let mut used_host_names = discovery.used_host_names;
        let mut device_params = Vec::new();
        let mut host_params = Vec::new();
        let mut device_call_args = Vec::new();
        let mut prelude = Vec::new();
        let mut host_prelude = Vec::new();
        let mut names = HashMap::new();
        let mut shared_indexing = HashMap::new();
        let mut static_view_ranks = HashMap::new();
        let mut emitted_extent_params = HashSet::new();
        let device_extent_names = device_extent_names(program);
        let mut element_count = None;
        let mut shared_layout = SharedMemoryLayout::new();

        // Emit the host launch ABI and the device kernel ABI in boundary order.
        // - Extents are either host launch inputs or compile-time macros from
        //   --cuda-static.
        // - Globals become device kernel pointers plus size values
        //   derived from their gpu.global dimensions.
        // - Shared memory is static when all of its dimensions are static; otherwise it slices CUDA's
        //   dynamic shared-memory buffer.
        for variable in boundary {
            if let Some(leaf) = extent_leaf(&variable.ty) {
                let Some(name) = discovery.extent_param_names.get(&leaf).cloned() else {
                    return Err(CudaAbiError::MissingGridExtent(leaf));
                };
                names.insert(variable.name.clone(), name.clone());
                if discovery.static_extent_leaves.contains(&leaf) {
                    continue;
                }
                if emitted_extent_params.insert(leaf) {
                    host_params.push(Param {
                        ty: "uint64_t".to_string(),
                        name,
                    });
                }
                if device_extent_names.contains(&variable.name) {
                    let device_name =
                        unique_name(&sanitize_ident(&variable.name), &mut used_device_names);
                    names.insert(variable.name.clone(), device_name.clone());
                    device_params.push(Param {
                        ty: "uint64_t".to_string(),
                        name: device_name,
                    });
                    device_call_args.push(
                        discovery
                            .extent_param_names
                            .get(&leaf)
                            .cloned()
                            .ok_or(CudaAbiError::MissingGridExtent(leaf))?,
                    );
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

            if let Some(shared) = gpu_shared(&variable.ty)? {
                let base_name = sanitize_ident(&variable.name);
                let device_name = unique_name(&base_name, &mut used_device_names);
                let memory = SharedMemory::from_gpu_shared(
                    &shared,
                    &discovery.extent_param_names,
                    &discovery.static_extent_leaves,
                )?;

                // Static shared memory has compile-time dimensions and becomes
                // a CUDA `__shared__` array. Dynamic shared memory keeps its
                // previous flat-buffer ABI and contributes to the launch byte
                // count.
                let binding = match memory {
                    SharedMemory::Static(memory) => bind_static_shared(device_name, memory),
                    SharedMemory::Dynamic(memory) => shared_layout.bind_dynamic(
                        device_name,
                        memory,
                        &mut used_device_names,
                        &mut used_host_names,
                    ),
                };

                names.insert(variable.name.clone(), binding.device_name.clone());
                shared_indexing.insert(binding.device_name.clone(), binding.indexing);
                device_params.extend(binding.device_params);
                host_prelude.extend(binding.host_prelude);
                device_call_args.extend(binding.device_call_args);
                prelude.extend(binding.device_prelude);
                continue;
            }

            // gpu.grid values are handled above as launch contracts and erased
            // from the CUDA parameter list.
            if gpu_grid(&variable.ty)?.is_some() {
                continue;
            }
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
        let dynamic_shared_bytes = shared_layout.dynamic_shared_bytes();
        collect_shared_aliases(&program.body, &names, &mut shared_indexing);
        collect_static_shared_views(
            &program.body,
            &names,
            &shared_indexing,
            &mut static_view_ranks,
        );

        Ok(Self {
            device_params,
            host_params,
            device_call_args,
            prelude,
            host_prelude,
            macros: discovery.macros,
            launch,
            dynamic_shared_bytes,
            shared_indexing,
            static_view_ranks,
            names,
        })
    }

    pub(super) fn rename(&self, name: &str) -> String {
        self.names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    pub(super) fn shared_access(&self, shared: &str, view: &str) -> String {
        match self.shared_indexing.get(shared) {
            Some(SharedIndexing::Static { rank: 1 }) => {
                format!("{shared}[{}]", view_component(view, "x"))
            }
            Some(SharedIndexing::Static { rank: 2 }) => {
                format!(
                    "{shared}[{}][{}]",
                    view_component(view, "x"),
                    view_component(view, "y")
                )
            }
            Some(SharedIndexing::Static { rank: 3 }) => {
                format!(
                    "{shared}[{}][{}][{}]",
                    view_component(view, "x"),
                    view_component(view, "y"),
                    view_component(view, "z")
                )
            }
            _ => format!("{shared}[{view}]"),
        }
    }

    pub(super) fn static_view_rank(&self, view: &str) -> Option<usize> {
        self.static_view_ranks.get(view).copied()
    }
}

fn view_component(view: &str, component: &str) -> String {
    format!("{view}_{component}")
}

fn collect_shared_aliases(
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    shared_indexing: &mut HashMap<String, SharedIndexing>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_shared_aliases(body, names, shared_indexing);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_shared_aliases(then_body, names, shared_indexing);
                collect_shared_aliases(else_body, names, shared_indexing);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_shared_aliases(case, names, shared_indexing);
                }
            }
            Stmt::Primitive(primitive) if primitive.name == "gpu.shared.store" => {
                let Some(shared) = primitive.inputs.first() else {
                    continue;
                };
                let Some(output) = primitive.outputs.first() else {
                    continue;
                };
                let shared = names.get(shared).unwrap_or(shared);
                let output = names.get(output).unwrap_or(output);
                if let Some(indexing) = shared_indexing.get(shared).cloned() {
                    shared_indexing.insert(output.clone(), indexing);
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

fn collect_static_shared_views(
    stmts: &[Stmt],
    names: &HashMap<String, String>,
    shared_indexing: &HashMap<String, SharedIndexing>,
    static_view_ranks: &mut HashMap<String, usize>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_static_shared_views(body, names, shared_indexing, static_view_ranks);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_static_shared_views(then_body, names, shared_indexing, static_view_ranks);
                collect_static_shared_views(else_body, names, shared_indexing, static_view_ranks);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_static_shared_views(case, names, shared_indexing, static_view_ranks);
                }
            }
            Stmt::Primitive(primitive)
                if primitive.name == "gpu.shared.load" || primitive.name == "gpu.shared.store" =>
            {
                let Some(shared) = primitive.inputs.first() else {
                    continue;
                };
                let Some(view) = primitive.inputs.get(1) else {
                    continue;
                };
                let shared = names.get(shared).unwrap_or(shared);
                if let Some(SharedIndexing::Static { rank }) = shared_indexing.get(shared) {
                    static_view_ranks.insert(view.clone(), *rank);
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

fn device_extent_names(program: &StructuredProgram) -> HashSet<String> {
    let mut names = HashSet::new();
    collect_device_extent_names(&program.body, &mut names);
    names
}

fn collect_device_extent_names(stmts: &[Stmt], names: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } | Stmt::Loop { body, .. } | Stmt::For { body, .. } => {
                collect_device_extent_names(body, names);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                collect_device_extent_names(then_body, names);
                collect_device_extent_names(else_body, names);
            }
            Stmt::Switch { cases, .. } => {
                for case in cases {
                    collect_device_extent_names(case, names);
                }
            }
            Stmt::Primitive(primitive) => collect_primitive_device_extents(primitive, names),
            Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::Assign { .. }
            | Stmt::Comment(_) => {}
        }
    }
}

fn collect_primitive_device_extents(primitive: &Primitive, names: &mut HashSet<String>) {
    if primitive.name == "gpu.view.group"
        && let Some(thread_count) = primitive.inputs.get(2)
    {
        names.insert(thread_count.clone());
    }
}

struct BoundaryDiscovery {
    grid_shape: GridShape,
    extent_param_names: HashMap<usize, String>,
    static_extent_leaves: HashSet<usize>,
    used_host_names: HashSet<String>,
    macros: Vec<CudaMacro>,
}

// Tracks the dynamic shared-memory portion while boundary values are emitted.
// Static gpu.shared values are emitted independently as `__shared__` arrays;
// dynamic values share CUDA's single `extern __shared__` buffer as adjacent
// slices.
struct SharedMemoryLayout {
    // Device-side size params are used in the kernel prelude to compute slice
    // offsets, e.g. `tile_b = __shared_mem + tile_a_size`.
    device_size_names: Vec<String>,
    // Host-side size locals are used to compute the third CUDA launch argument,
    // e.g. `<<<grid, block, (tile_a_size + tile_b_size) * sizeof(float)>>>`.
    host_size_names: Vec<String>,
}

struct SharedBinding {
    device_name: String,
    indexing: SharedIndexing,
    device_params: Vec<Param>,
    host_prelude: Vec<String>,
    device_call_args: Vec<String>,
    device_prelude: Vec<String>,
}

struct StaticSharedMemory {
    element_ty: &'static str,
    dimensions: Vec<DimensionExpr>,
}

struct DynamicSharedMemory {
    element_ty: &'static str,
    size_expr: String,
}

enum SharedMemory {
    Static(StaticSharedMemory),
    Dynamic(DynamicSharedMemory),
}

impl SharedMemory {
    fn from_gpu_shared(
        shared: &GpuShared<'_>,
        extent_param_names: &HashMap<usize, String>,
        static_extent_leaves: &HashSet<usize>,
    ) -> Result<Self, CudaAbiError> {
        let element_ty = cuda_shared_element_type(shared)?;
        let size = shared_size_expr(shared, extent_param_names, static_extent_leaves)?;

        if size.is_static {
            Ok(Self::Static(StaticSharedMemory {
                element_ty,
                dimensions: size.dimensions,
            }))
        } else {
            Ok(Self::Dynamic(DynamicSharedMemory {
                element_ty,
                size_expr: size.expr,
            }))
        }
    }
}

impl SharedMemoryLayout {
    fn new() -> Self {
        Self {
            device_size_names: Vec::new(),
            host_size_names: Vec::new(),
        }
    }

    fn bind_dynamic(
        &mut self,
        device_name: String,
        memory: DynamicSharedMemory,
        used_device_names: &mut HashSet<String>,
        used_host_names: &mut HashSet<String>,
    ) -> SharedBinding {
        let device_size_name = unique_name(&format!("{device_name}_size"), used_device_names);
        let host_size_name = unique_name(&format!("{device_name}_size"), used_host_names);
        let offset_expr = self.device_offset_expr();

        // The host computes each shared allocation size from the original extent
        // parameters, then passes that size into the device kernel. The device
        // uses those size params only for pointer arithmetic between slices.
        let mut device_prelude = Vec::new();
        if self.device_size_names.is_empty() {
            // First shared value starts at the beginning of the CUDA dynamic
            // shared-memory region.
            device_prelude.push(format!(
                "extern __shared__ {} __shared_mem[];",
                memory.element_ty
            ));
            device_prelude.push(format!(
                "{}* {device_name} = __shared_mem;",
                memory.element_ty
            ));
        } else {
            // Later shared values start after all previously bound slices.
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

    fn dynamic_shared_bytes(&self) -> Option<String> {
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

fn bind_static_shared(device_name: String, memory: StaticSharedMemory) -> SharedBinding {
    // Static shared memory is emitted as a normal CUDA `__shared__` allocation.
    // It has no host launch byte count and no device size parameter because all
    // dimensions are compile-time expressions.
    SharedBinding {
        device_name: device_name.clone(),
        indexing: SharedIndexing::Static {
            rank: memory.dimensions.len(),
        },
        device_params: Vec::new(),
        host_prelude: Vec::new(),
        device_call_args: Vec::new(),
        device_prelude: vec![format!(
            "__shared__ {} {device_name}{};",
            memory.element_ty,
            cuda_array_dimensions(&memory.dimensions)
        )],
    }
}

fn discover_boundary(
    boundary: &[&Variable],
    options: &CudaOptions,
) -> Result<BoundaryDiscovery, CudaAbiError> {
    let mut grid_shape = None;
    let mut extent_param_names = HashMap::new();
    let mut static_extent_leaves = HashSet::new();
    let mut used_host_names = HashSet::new();
    let mut used_macro_names = HashSet::new();
    let mut macros = Vec::new();
    let mut seen_static_names = HashSet::new();

    // This pass only discovers facts needed before ABI emission. In particular,
    // gpu.global sizes may reference extent leaves that appear anywhere in the
    // boundary, so extent names must be known before globals are lowered.
    for variable in boundary {
        if let Some(shape) = GridShape::from_type(&variable.ty)? {
            if grid_shape.replace(shape).is_some() {
                return Err(CudaAbiError::DuplicateGrid);
            }
        }

        let requested_static = options.static_values.get(&variable.name).copied();
        if requested_static.is_some() {
            seen_static_names.insert(variable.name.clone());
        }

        // gpu.grid, gpu.global, and gpu.shared dimensions refer to extent
        // leaves. A leaf resolves either to a host parameter or, when the user
        // provided --cuda-static for that boundary name, to a preprocessor
        // macro that specializes the generated CUDA.
        if let Some(leaf) = extent_leaf(&variable.ty) {
            let name = if let Some(value) = requested_static {
                let name = unique_name(&macro_ident(&variable.name), &mut used_macro_names);
                macros.push(CudaMacro {
                    name: name.clone(),
                    value,
                });
                static_extent_leaves.insert(leaf);
                name
            } else {
                unique_name(&sanitize_ident(&variable.name), &mut used_host_names)
            };
            extent_param_names.insert(leaf, name);
        } else if requested_static.is_some() {
            return Err(CudaAbiError::StaticValueNotExtent(variable.name.clone()));
        }
    }

    for name in options.static_values.keys() {
        if !seen_static_names.contains(name) {
            return Err(CudaAbiError::UnknownStaticValue(name.clone()));
        }
    }

    Ok(BoundaryDiscovery {
        grid_shape: grid_shape.ok_or(CudaAbiError::MissingGrid)?,
        extent_param_names,
        static_extent_leaves,
        used_host_names,
        macros,
    })
}

#[derive(Debug, Clone)]
struct GridLaunch {
    grid: Vec<String>,
    block: Vec<String>,
}

#[derive(Debug, Clone)]
struct GridShape {
    grid: Vec<Obj>,
    block: Vec<Obj>,
}

impl GridShape {
    fn from_type(obj: &Obj) -> Result<Option<Self>, CudaAbiError> {
        let Some(gpu_grid) = gpu_grid(obj)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            grid: dimension_terms(gpu_grid.grid).ok_or(CudaAbiError::InvalidGridShape)?,
            block: dimension_terms(gpu_grid.block).ok_or(CudaAbiError::InvalidGridShape)?,
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
    dimensions: &[Obj],
    extent_param_names: &HashMap<usize, String>,
) -> Result<Vec<String>, CudaAbiError> {
    dimensions
        .iter()
        .map(|dimension| {
            dimension_expr(
                dimension,
                extent_param_names,
                CudaAbiError::MissingGridExtent,
                || CudaAbiError::InvalidGridShape,
            )
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

fn dimension_terms(dimension: &Obj) -> Option<Vec<Obj>> {
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
    Some(children.clone())
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

fn macro_ident(name: &str) -> String {
    let mut ident = sanitize_ident(name).to_ascii_uppercase();
    if ident.is_empty() {
        ident.push_str("STATIC_VALUE");
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

fn cuda_shared_element_type(shared: &GpuShared<'_>) -> Result<&'static str, CudaAbiError> {
    match shared.element {
        "f32" => Ok("float"),
        _ => Err(CudaAbiError::UnsupportedSharedElement(
            shared.element.to_string(),
        )),
    }
}

#[derive(Debug, Clone)]
struct GpuGlobal<'a> {
    element: &'a str,
    dimensions: Vec<&'a Obj>,
}

#[derive(Debug, Clone)]
struct GpuShared<'a> {
    element: &'a str,
    dimensions: Vec<&'a Obj>,
}

fn gpu_global(obj: &Obj) -> Result<Option<GpuGlobal<'_>>, CudaAbiError> {
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

fn gpu_shared(obj: &Obj) -> Result<Option<GpuShared<'_>>, CudaAbiError> {
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

fn global_size_expr(
    global: &GpuGlobal<'_>,
    extent_param_names: &HashMap<usize, String>,
) -> Result<String, CudaAbiError> {
    let dimensions = global
        .dimensions
        .iter()
        .map(|dimension| {
            dimension_expr(
                dimension,
                extent_param_names,
                CudaAbiError::MissingGlobalExtent,
                || CudaAbiError::InvalidGlobalShape,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(dimensions.join(" * "))
}

fn shared_size_expr(
    shared: &GpuShared<'_>,
    extent_param_names: &HashMap<usize, String>,
    static_extent_leaves: &HashSet<usize>,
) -> Result<SharedSizeExpr, CudaAbiError> {
    let dimensions = shared
        .dimensions
        .iter()
        .map(|dimension| {
            dimension_expr_with_static(
                dimension,
                extent_param_names,
                static_extent_leaves,
                CudaAbiError::MissingSharedExtent,
                || CudaAbiError::InvalidSharedShape,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SharedSizeExpr {
        expr: product_expr(&dimensions),
        is_static: dimensions.iter().all(|dimension| dimension.is_static),
        dimensions,
    })
}

#[derive(Debug, Clone)]
struct SharedSizeExpr {
    expr: String,
    is_static: bool,
    dimensions: Vec<DimensionExpr>,
}

#[derive(Debug, Clone)]
struct DimensionExpr {
    expr: String,
    is_static: bool,
}

fn product_expr(dimensions: &[DimensionExpr]) -> String {
    dimensions
        .iter()
        .map(|dimension| dimension.expr.as_str())
        .collect::<Vec<_>>()
        .join(" * ")
}

fn cuda_array_dimensions(dimensions: &[DimensionExpr]) -> String {
    dimensions
        .iter()
        .map(|dimension| format!("[{}]", dimension.expr))
        .collect::<String>()
}

fn dimension_expr(
    dimension: &Obj,
    extent_param_names: &HashMap<usize, String>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<String, CudaAbiError> {
    match dimension {
        Tree::Leaf(leaf, _) => extent_param_names
            .get(leaf)
            .cloned()
            .ok_or_else(|| missing_extent(*leaf)),
        Tree::Node(op, 0, children)
            if matches!(op.to_string().as_str(), "nat.mul" | "*") && children.len() == 2 =>
        {
            let lhs = dimension_expr(
                &children[0],
                extent_param_names,
                missing_extent,
                invalid_shape,
            )?;
            let rhs = dimension_expr(
                &children[1],
                extent_param_names,
                missing_extent,
                invalid_shape,
            )?;
            Ok(format!("({lhs} * {rhs})"))
        }
        Tree::Node(op, 0, children) if op.to_string() == "nat.ceil-div" && children.len() == 2 => {
            let numerator = dimension_expr(
                &children[0],
                extent_param_names,
                missing_extent,
                invalid_shape,
            )?;
            let denominator = dimension_expr(
                &children[1],
                extent_param_names,
                missing_extent,
                invalid_shape,
            )?;
            Ok(format!(
                "(({numerator} + {denominator} - 1) / {denominator})"
            ))
        }
        _ => {
            // TODO: allow literal dimension values. At the moment dimensions
            // must be expressed in terms of extent-backed hypergraph leaves and
            // supported nat operations.
            Err(invalid_shape())
        }
    }
}

fn dimension_expr_with_static(
    dimension: &Obj,
    extent_param_names: &HashMap<usize, String>,
    static_extent_leaves: &HashSet<usize>,
    missing_extent: impl Fn(usize) -> CudaAbiError + Copy,
    invalid_shape: impl Fn() -> CudaAbiError + Copy,
) -> Result<DimensionExpr, CudaAbiError> {
    match dimension {
        Tree::Leaf(leaf, _) => Ok(DimensionExpr {
            expr: extent_param_names
                .get(leaf)
                .cloned()
                .ok_or_else(|| missing_extent(*leaf))?,
            is_static: static_extent_leaves.contains(leaf),
        }),
        Tree::Node(op, 0, children)
            if matches!(op.to_string().as_str(), "nat.mul" | "*") && children.len() == 2 =>
        {
            let lhs = dimension_expr_with_static(
                &children[0],
                extent_param_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            let rhs = dimension_expr_with_static(
                &children[1],
                extent_param_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            Ok(DimensionExpr {
                expr: format!("({} * {})", lhs.expr, rhs.expr),
                is_static: lhs.is_static && rhs.is_static,
            })
        }
        Tree::Node(op, 0, children) if op.to_string() == "nat.ceil-div" && children.len() == 2 => {
            let numerator = dimension_expr_with_static(
                &children[0],
                extent_param_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            let denominator = dimension_expr_with_static(
                &children[1],
                extent_param_names,
                static_extent_leaves,
                missing_extent,
                invalid_shape,
            )?;
            Ok(DimensionExpr {
                expr: format!(
                    "(({} + {} - 1) / {})",
                    numerator.expr, denominator.expr, denominator.expr
                ),
                is_static: numerator.is_static && denominator.is_static,
            })
        }
        _ => {
            // TODO: allow literal dimension values and mark them static. Today
            // static shared memory is driven by --cuda-static extent names.
            Err(invalid_shape())
        }
    }
}

fn memory_dimensions(dimensions: &Obj) -> Option<Vec<&Obj>> {
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
