use crate::{
    compile::{
        cuda::{
            CudaOptions,
            abi::{CudaAbiError, CudaKernelAbi},
            render::{CudaPrimitiveLowering, render_cuda},
        },
        program::Definition,
        proof::ProofEvidence,
    },
    structured::ir::{Primitive, StructuredProgram},
};
use hexpr::Operation;
use metacat::theory::{Theory, TheoryId, TheorySet};

#[derive(Debug, Clone)]
pub(super) struct CudaTarget<'a> {
    abi: CudaKernelAbi,
    primitives: GenericCudaPrimitives<'a>,
}

impl<'a> CudaTarget<'a> {
    pub(super) fn new(
        theory_set: &'a TheorySet,
        entry: &Definition,
        program: &StructuredProgram,
        options: &CudaOptions,
        proof_evidence: Option<&ProofEvidence>,
    ) -> Result<Self, CudaAbiError> {
        Ok(Self {
            abi: CudaKernelAbi::from_definition(entry, program, options, proof_evidence)?,
            primitives: GenericCudaPrimitives::new(theory_set),
        })
    }

    pub(super) fn render_cuda_with_launch(&self, program: &StructuredProgram) -> String {
        render_cuda(program, &self.abi, &self.primitives)
    }
}

#[derive(Debug, Clone, Copy)]
struct GenericCudaPrimitives<'a> {
    data_theory: Option<&'a Theory>,
    f32: F32Primitives,
    gpu: GpuPrimitives,
}

impl<'a> GenericCudaPrimitives<'a> {
    fn new(theory_set: &'a TheorySet) -> Self {
        Self {
            data_theory: theory(theory_set, "data"),
            f32: F32Primitives,
            gpu: GpuPrimitives,
        }
    }

    fn expand_data_arrow(&self, local_name: &str) -> Option<Vec<String>> {
        let data_theory = self.data_theory?;
        let mut stack = Vec::new();
        self.expand_local_data_arrow(data_theory, local_name, 0, &mut stack)
    }

    fn expand_local_data_arrow(
        &self,
        data_theory: &Theory,
        local_name: &str,
        depth: usize,
        stack: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        const MAX_EXPANSION_DEPTH: usize = 8;

        let op = parse_operation(local_name)?;
        let arrow = data_theory.get_arrow(&op)?;
        let definition = arrow.definition.as_ref()?;

        let qualified = format!("data.{local_name}");
        if stack.iter().any(|entry| entry == &qualified) {
            return Some(vec![format!(
                "/* TODO: recursive CUDA expansion stopped at Catena arrow `{qualified}` */"
            )]);
        }
        if depth >= MAX_EXPANSION_DEPTH {
            return Some(vec![format!(
                "/* TODO: CUDA expansion depth limit reached at Catena arrow `{qualified}` */"
            )]);
        }

        stack.push(qualified.clone());
        let mut lines = vec![format!("/* begin expanded Catena arrow `{qualified}` */")];
        for edge in &definition.hypergraph.edges {
            let edge_name = edge.to_string();
            if let Some(mut nested) =
                self.expand_local_data_arrow(data_theory, &edge_name, depth + 1, stack)
            {
                indent_expansion(&mut nested);
                lines.extend(nested);
            } else {
                lines.push(format!(
                    "  /* TODO: no CUDA lowering for Catena arrow `{edge_name}` */"
                ));
            }
        }
        lines.push(format!("/* end expanded Catena arrow `{qualified}` */"));
        stack.pop();
        Some(lines)
    }
}

impl CudaPrimitiveLowering for GenericCudaPrimitives<'_> {
    fn lower_primitive_lines(&self, primitive: &Primitive, abi: &CudaKernelAbi) -> Vec<String> {
        let Some((namespace, local_name)) = primitive.name.split_once('.') else {
            return fallback_primitive_lines(primitive);
        };

        let local = PrimitiveLocalName::new(local_name);
        let lines = match namespace {
            "data" => self.expand_data_arrow(local_name),
            "f32" => self.f32.lower(&local, primitive, abi),
            "gpu" => self.gpu.lower(&local, primitive, abi),
            _ => None,
        };

        if let Some(lines) = lines {
            return lines;
        }

        fallback_primitive_lines(primitive)
    }
}

#[derive(Debug, Clone)]
struct PrimitiveLocalName<'a> {
    segments: Vec<&'a str>,
}

impl<'a> PrimitiveLocalName<'a> {
    fn new(name: &'a str) -> Self {
        let segments = name.split('.').collect();
        Self { segments }
    }

    fn matches(&self, segments: &[&str]) -> bool {
        self.segments == segments
    }
}

trait NamespaceLowering {
    fn lower(
        &self,
        local: &PrimitiveLocalName<'_>,
        primitive: &Primitive,
        abi: &CudaKernelAbi,
    ) -> Option<Vec<String>>;
}

#[derive(Debug, Clone, Copy)]
struct F32Primitives;

impl NamespaceLowering for F32Primitives {
    fn lower(
        &self,
        local: &PrimitiveLocalName<'_>,
        primitive: &Primitive,
        _abi: &CudaKernelAbi,
    ) -> Option<Vec<String>> {
        let [out] = primitive.outputs.as_slice() else {
            return None;
        };

        if local.matches(&["one"]) {
            return Some(vec![format!("float {out} = 1.0f;")]);
        }
        if local.matches(&["zero"]) {
            return Some(vec![format!("float {out} = 0.0f;")]);
        }
        if local.matches(&["add"]) {
            let [lhs, rhs] = primitive.inputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("float {out} = {lhs} + {rhs};")]);
        }

        None
    }
}

#[derive(Debug, Clone, Copy)]
struct GpuPrimitives;

impl NamespaceLowering for GpuPrimitives {
    fn lower(
        &self,
        local: &PrimitiveLocalName<'_>,
        primitive: &Primitive,
        abi: &CudaKernelAbi,
    ) -> Option<Vec<String>> {
        if local.matches(&["grid", "view"]) {
            let [_grid] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let mut lines = vec![
                format!("uint3 {out}_block = blockIdx;"),
                format!("uint3 {out}_thread = threadIdx;"),
            ];
            if abi.static_view_rank(out).is_some() {
                lines.extend(view_coordinate_lines_from_grid_view(out, out));
            }
            return Some(lines);
        }

        if local.matches(&["view", "linearize"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let lines = if abi.is_grid_view(view) {
                vec![format!(
                    "uint64_t {out} = (uint64_t){} * blockDim.x + {};",
                    view_block_component(view, "x"),
                    view_thread_component(view, "x")
                )]
            } else {
                vec![format!("uint64_t {out} = {view};")]
            };
            return Some(lines);
        }

        if local.matches(&["view", "prev"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("uint64_t {out} = {view} - 1;")]);
        }

        if local.matches(&["view", "prev2"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("uint64_t {out} = {view} - 2;")]);
        }

        if local.matches(&["view", "is-zero"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("bool {out} = {view} == 0;")]);
        }

        if local.matches(&["view", "is-one"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("bool {out} = {view} == 1;")]);
        }

        if local.matches(&["view", "group-by-tile"]) {
            let [view, tile_rows, tile_cols] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let mut lines = if abi.is_grid_view(view) {
                vec![
                    format!(
                        "uint64_t {out}_row = (uint64_t){} * {tile_rows} + {};",
                        view_block_component(view, "y"),
                        view_thread_component(view, "y")
                    ),
                    format!(
                        "uint64_t {out}_col = (uint64_t){} * {tile_cols} + {};",
                        view_block_component(view, "x"),
                        view_thread_component(view, "x")
                    ),
                ]
            } else {
                vec![
                    format!("uint64_t {out}_row = {view} / {tile_cols};"),
                    format!("uint64_t {out}_col = {view} % {tile_cols};"),
                ]
            };
            if abi.static_view_rank(out).is_some() && abi.is_grid_view(view) {
                lines.extend(view_coordinate_lines_from_grid_view(out, view));
            }
            return Some(lines);
        }

        if local.matches(&["view", "group"]) {
            let [view, cols] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let lines = vec![
                format!("uint64_t {out}_row = {view} / {cols};"),
                format!("uint64_t {out}_col = {view} % {cols};"),
            ];
            return Some(lines);
        }

        if local.matches(&["view", "reshape"]) {
            let [view, shape] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let shape_value = abi.shape_value(shape);
            let rows = shape_value
                .and_then(|value| value.first())
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            let cols = shape_value
                .and_then(|value| value.get(1))
                .cloned()
                .unwrap_or_else(|| "1".to_string());
            let row = if rows == "1" {
                "0".to_string()
            } else if cols == "1" {
                view.to_string()
            } else {
                format!("{view} / {cols}")
            };
            let col = if cols == "1" {
                "0".to_string()
            } else if rows == "1" {
                view.to_string()
            } else {
                format!("{view} % {cols}")
            };
            let lines = vec![
                format!("uint64_t {out}_row = {row};"),
                format!("uint64_t {out}_col = {col};"),
            ];
            return Some(lines);
        }

        if local.matches(&["shape", "row"])
            || local.matches(&["shape", "row-mul"])
            || local.matches(&["shape", "col"])
            || local.matches(&["shape", "col-mul"])
            || local.matches(&["shape", "2d"])
        {
            return Some(Vec::new());
        }

        if local.matches(&["view", "row"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("uint64_t {out} = {view}_row;")]);
        }

        if local.matches(&["view", "col"]) {
            let [view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!("uint64_t {out} = {view}_col;")]);
        }

        if local.matches(&["global", "store"]) {
            let [global, view, value] = primitive.inputs.as_slice() else {
                return None;
            };
            let output = primitive.outputs.first();
            let mut lines = certified_access_lines(abi, global, view);
            lines.push(format!("{} = {value};", abi.global_access(global, view)));
            if let Some(output) = output
                && output != global
            {
                lines.push(format!("auto {output} = {global};"));
            }
            return Some(lines);
        }

        if local.matches(&["global", "load"]) {
            let [global, view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let mut lines = certified_access_lines(abi, global, view);
            lines.push(format!(
                "float {out} = {};",
                abi.global_access(global, view)
            ));
            return Some(lines);
        }

        if local.matches(&["shared", "load"]) {
            let [shared, view] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            let mut lines = certified_access_lines(abi, shared, view);
            lines.push(format!(
                "float {out} = {};",
                abi.shared_access(shared, view)
            ));
            return Some(lines);
        }

        if local.matches(&["shared", "store"]) {
            let [shared, view, value] = primitive.inputs.as_slice() else {
                return None;
            };
            let output = primitive.outputs.first();
            let mut lines = certified_access_lines(abi, shared, view);
            lines.push(format!("{} = {value};", abi.shared_access(shared, view)));
            if let Some(output) = output
                && output != shared
            {
                lines.push(format!("auto {output} = {shared};"));
            }
            return Some(lines);
        }

        None
    }
}

fn view_coordinate_lines_from_grid_view(view: &str, grid_view: &str) -> Vec<String> {
    vec![
        format!(
            "uint64_t {view}_x = {};",
            view_thread_component(grid_view, "x")
        ),
        format!(
            "uint64_t {view}_y = {};",
            view_thread_component(grid_view, "y")
        ),
        format!(
            "uint64_t {view}_z = {};",
            view_thread_component(grid_view, "z")
        ),
    ]
}

fn view_block_component(view: &str, component: &str) -> String {
    format!("{view}_block.{component}")
}

fn view_thread_component(view: &str, component: &str) -> String {
    format!("{view}_thread.{component}")
}

fn certified_access_lines(abi: &CudaKernelAbi, memory: &str, view: &str) -> Vec<String> {
    let Some(certificate) = abi.access_certificate(memory, view) else {
        return Vec::new();
    };
    vec![format!("// safety: certified by {certificate}")]
}

fn fallback_primitive_lines(primitive: &Primitive) -> Vec<String> {
    vec![format!(
        "/* TODO: lower Catena primitive `{}` as `{}` */",
        primitive.name,
        primitive_assignment(primitive)
    )]
}

fn primitive_assignment(primitive: &Primitive) -> String {
    let call = format!("{}({})", primitive.name, primitive.inputs.join(", "));
    if primitive.outputs.is_empty() {
        call
    } else {
        format!("{} = {call}", primitive.outputs.join(", "))
    }
}

fn theory<'a>(theory_set: &'a TheorySet, name: &str) -> Option<&'a Theory> {
    let id = TheoryId(parse_operation(name)?);
    theory_set.theories.get(&id)
}

fn parse_operation(name: &str) -> Option<Operation> {
    name.parse().ok()
}

fn indent_expansion(lines: &mut [String]) {
    for line in lines {
        line.insert_str(0, "  ");
    }
}
