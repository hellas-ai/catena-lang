use crate::{
    compile::{
        cuda::{
            abi::CudaKernelAbi,
            render::{CudaPrimitiveLowering, render_cuda},
        },
        program::Definition,
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
    pub(super) fn new(theory_set: &'a TheorySet, entry: &Definition) -> Self {
        Self {
            abi: CudaKernelAbi::from_definition(entry),
            primitives: GenericCudaPrimitives::new(theory_set),
        }
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
    fn lower_primitive_lines(&self, primitive: &Primitive) -> Vec<String> {
        let Some((namespace, local_name)) = primitive.name.split_once('.') else {
            return fallback_primitive_lines(primitive);
        };

        let local = PrimitiveLocalName::new(local_name);
        let lines = match namespace {
            "data" => self.expand_data_arrow(local_name),
            "f32" => self.f32.lower(&local, primitive),
            "gpu" => self.gpu.lower(&local, primitive),
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
    fn lower(&self, local: &PrimitiveLocalName<'_>, primitive: &Primitive) -> Option<Vec<String>>;
}

#[derive(Debug, Clone, Copy)]
struct F32Primitives;

impl NamespaceLowering for F32Primitives {
    fn lower(&self, local: &PrimitiveLocalName<'_>, primitive: &Primitive) -> Option<Vec<String>> {
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
    fn lower(&self, local: &PrimitiveLocalName<'_>, primitive: &Primitive) -> Option<Vec<String>> {
        if local.matches(&["view", "group"]) {
            let [block, thread, _block_size] = primitive.inputs.as_slice() else {
                return None;
            };
            let [out] = primitive.outputs.as_slice() else {
                return None;
            };
            return Some(vec![format!(
                "uint64_t {out} = (uint64_t){block}.x * blockDim.x + {thread}.x;"
            )]);
        }

        if local.matches(&["global", "store"]) {
            let [global, view, value] = primitive.inputs.as_slice() else {
                return None;
            };
            let output = primitive.outputs.first();
            let mut lines = vec![
                format!("if ({view} < __elements) {{"),
                format!("    {global}[{view}] = {value};"),
                "}".to_string(),
            ];
            if let Some(output) = output
                && output != global
            {
                lines.push(format!("auto {output} = {global};"));
            }
            return Some(lines);
        }

        None
    }
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
