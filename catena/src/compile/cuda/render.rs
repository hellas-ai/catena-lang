use crate::{
    compile::cuda::abi::CudaKernelAbi,
    structured::ir::{Primitive, Stmt, StructuredProgram},
};

pub(super) trait CudaPrimitiveLowering {
    fn lower_primitive_lines(&self, primitive: &Primitive) -> Vec<String>;
}

pub(super) fn render_cuda(
    program: &StructuredProgram,
    abi: &CudaKernelAbi,
    primitives: &impl CudaPrimitiveLowering,
) -> String {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n\n");
    out.push_str(&format!("__global__ void {}(", program.entry.name));
    out.push_str(
        &abi.params
            .iter()
            .map(|p| format!("{} {}", p.ty, p.name))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    render_prelude(&mut out, abi);
    render_cuda_stmts(&mut out, &program.body, 1, abi, primitives);
    out.push_str("}\n\n");
    render_launch_helper(&mut out, program, abi);
    out
}

fn render_prelude(out: &mut String, abi: &CudaKernelAbi) {
    for line in &abi.prelude {
        out.push_str(&format!("    {line}\n"));
    }
    if let Some(element_count) = &abi.launch.element_count {
        out.push_str(&format!("    uint64_t __elements = {element_count};\n"));
    }
    if !abi.prelude.is_empty() || abi.launch.element_count.is_some() {
        out.push('\n');
    }
}

fn render_launch_helper(out: &mut String, program: &StructuredProgram, abi: &CudaKernelAbi) {
    out.push_str(&format!("void launch_{}(", program.entry.name));
    out.push_str(
        &abi.params
            .iter()
            .map(|p| format!("{} {}", p.ty, p.name))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    render_launch_config(out, abi);
    out.push_str(&format!(
        "    {}<<<grid, block>>>({});\n",
        program.entry.name,
        abi.params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str("}\n");
}

fn render_launch_config(out: &mut String, abi: &CudaKernelAbi) {
    out.push_str(&format!("    dim3 block({});\n", abi.launch.block_expr));
    out.push_str(&format!("    dim3 grid({});\n", abi.launch.grid_expr));
}

fn render_cuda_stmts(
    out: &mut String,
    stmts: &[Stmt],
    indent: usize,
    abi: &CudaKernelAbi,
    domain: &impl CudaPrimitiveLowering,
) {
    let pad = "    ".repeat(indent);
    for stmt in stmts {
        match stmt {
            Stmt::Block { body, .. } => {
                out.push_str(&format!("{pad}do {{\n"));
                render_cuda_stmts(out, body, indent + 1, abi, domain);
                out.push_str(&format!("{pad}}} while (0);\n"));
            }
            Stmt::Loop { body, .. } => {
                out.push_str(&format!("{pad}while (1) {{\n"));
                render_cuda_stmts(out, body, indent + 1, abi, domain);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::For {
                var, extent, body, ..
            } => {
                out.push_str(&format!(
                    "{pad}for (int {var} = 0; {var} < {extent}; ++{var}) {{\n"
                ));
                render_cuda_stmts(out, body, indent + 1, abi, domain);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::If {
                condition,
                then_body,
                else_body,
            } => {
                out.push_str(&format!("{pad}if ({condition}) {{\n"));
                render_cuda_stmts(out, then_body, indent + 1, abi, domain);
                out.push_str(&format!("{pad}}} else {{\n"));
                render_cuda_stmts(out, else_body, indent + 1, abi, domain);
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Switch { selector, cases } => {
                out.push_str(&format!("{pad}switch ({selector}) {{\n"));
                for (index, body) in cases.iter().enumerate() {
                    out.push_str(&format!("{pad}case {index}:\n"));
                    render_cuda_stmts(out, body, indent + 1, abi, domain);
                    out.push_str(&format!("{pad}    break;\n"));
                }
                out.push_str(&format!("{pad}}}\n"));
            }
            Stmt::Break(label) => out.push_str(&format!("{pad}goto {label}_after;\n")),
            Stmt::Continue(label) => out.push_str(&format!("{pad}goto {label}_continue;\n")),
            Stmt::Return => out.push_str(&format!("{pad}return;\n")),
            Stmt::Barrier => out.push_str(&format!("{pad}__syncthreads();\n")),
            Stmt::Assign { lhs, rhs } => out.push_str(&format!("{pad}{lhs} = {rhs};\n")),
            Stmt::Primitive(primitive) => {
                let primitive = rename_primitive(primitive, abi);
                for line in domain.lower_primitive_lines(&primitive) {
                    out.push_str(&format!("{pad}{line}\n"));
                }
            }
            Stmt::Comment(comment) => out.push_str(&format!("{pad}// {comment}\n")),
        }
    }
}

fn rename_primitive(primitive: &Primitive, abi: &CudaKernelAbi) -> Primitive {
    Primitive {
        name: primitive.name.clone(),
        inputs: primitive
            .inputs
            .iter()
            .map(|name| abi.rename(name))
            .collect(),
        outputs: primitive
            .outputs
            .iter()
            .map(|name| abi.rename(name))
            .collect(),
        code: primitive.code.clone(),
    }
}
