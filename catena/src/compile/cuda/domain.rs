use crate::{
    compile::cuda::render::{CudaKernelAbi, CudaPrimitiveLowering, render_cuda},
    structured::{
        ir::{EntryPoint, Primitive, Program, Stmt},
        ramsey::ArrowSemantics,
    },
};
use hexpr::Operation;
use metacat::theory::{Theory, TheoryId, TheorySet};

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaTarget<'a> {
    pub(super) control: GenericCudaControl,
    abi: CudaKernelAbi,
    primitives: GenericCudaPrimitives<'a>,
}

impl<'a> CudaTarget<'a> {
    pub(super) fn new(theory_set: &'a TheorySet) -> Self {
        Self {
            control: GenericCudaControl,
            abi: CudaKernelAbi::Unknown,
            primitives: GenericCudaPrimitives::new(theory_set),
        }
    }

    pub(super) fn program(&self, entry: &str, body: Vec<Stmt>) -> Program {
        Program {
            name: sanitize_ident(entry),
            entry: EntryPoint {
                name: sanitize_ident(entry),
                params: Vec::new(),
            },
            body,
        }
    }

    pub(super) fn render_cuda_with_launch(&self, program: &Program) -> String {
        render_cuda(program, self.abi, &self.primitives)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GenericCudaControl;

impl ArrowSemantics for GenericCudaControl {
    fn actions(&self, op: &str) -> Vec<Stmt> {
        if op == "gpu.sync" {
            return vec![Stmt::Barrier];
        }
        vec![Stmt::Primitive(Primitive {
            name: op.to_string(),
            code: String::new(),
        })]
    }

    fn condition(&self, op: &str) -> String {
        format!("/* TODO: no CUDA condition lowering for Catena arrow `{op}` */ 1")
    }
}

#[derive(Debug, Clone, Copy)]
struct GenericCudaPrimitives<'a> {
    data_theory: Option<&'a Theory>,
}

impl<'a> GenericCudaPrimitives<'a> {
    fn new(theory_set: &'a TheorySet) -> Self {
        Self {
            data_theory: theory(theory_set, "data"),
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
        if let Some(local_name) = primitive.name.strip_prefix("data.") {
            if let Some(lines) = self.expand_data_arrow(local_name) {
                return lines;
            }
        }
        vec![format!(
            "/* TODO: no CUDA lowering for Catena arrow `{}` */",
            primitive.name
        )]
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
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
