use crate::{
    compile::cuda::render::{CudaKernelAbi, CudaPrimitiveLowering, render_cuda},
    structured::{
        ir::{EntryPoint, Primitive, Program, Stmt},
        ramsey::ArrowSemantics,
    },
};

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaTarget {
    pub(super) control: GenericCudaControl,
    abi: CudaKernelAbi,
    primitives: GenericCudaPrimitives,
}

impl CudaTarget {
    pub(super) fn new() -> Self {
        Self {
            control: GenericCudaControl,
            abi: CudaKernelAbi::Unknown,
            primitives: GenericCudaPrimitives,
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
        render_cuda(program, self.abi, self.primitives)
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
struct GenericCudaPrimitives;

impl CudaPrimitiveLowering for GenericCudaPrimitives {
    fn lower_primitive_lines(&self, primitive: &Primitive) -> Vec<String> {
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
