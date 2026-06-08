use crate::{
    compile::cuda::abi::CudaKernelAbi,
    structured::ir::{Primitive, Stmt, StructuredProgram},
};

pub(super) trait CudaPrimitiveLowering {
    fn lower_primitive_lines(&self, primitive: &Primitive, abi: &CudaKernelAbi) -> Vec<String>;
}

pub(super) fn render_cuda(
    program: &StructuredProgram,
    abi: &CudaKernelAbi,
    primitives: &impl CudaPrimitiveLowering,
) -> String {
    let mut out = String::new();
    out.push_str("#include <stdint.h>\n\n");
    render_macros(&mut out, abi);
    out.push_str(&format!("__global__ void {}(", program.entry.name));
    out.push_str(
        &abi.kernel_params
            .iter()
            .map(|p| format!("{} {}", p.ty, abi.annotated_name(&p.name)))
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

fn render_macros(out: &mut String, abi: &CudaKernelAbi) {
    for macro_def in &abi.macros {
        out.push_str(&format!("#ifndef {}\n", macro_def.name));
        out.push_str(&format!("#define {} {}\n", macro_def.name, macro_def.value));
        out.push_str("#endif\n");
    }
    if !abi.macros.is_empty() {
        out.push('\n');
    }
}

fn render_prelude(out: &mut String, abi: &CudaKernelAbi) {
    for line in &abi.kernel_prelude {
        out.push_str(&format!("    {line}\n"));
    }
    if !abi.kernel_prelude.is_empty() {
        out.push('\n');
    }
}

fn render_launch_helper(out: &mut String, program: &StructuredProgram, abi: &CudaKernelAbi) {
    out.push_str(&format!("void launch_{}(", program.entry.name));
    out.push_str(
        &abi.launcher_params
            .iter()
            .map(|p| format!("{} {}", p.ty, p.name))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    for line in &abi.launcher_prelude {
        out.push_str(&format!("    {line}\n"));
    }
    if !abi.launcher_prelude.is_empty() {
        out.push('\n');
    }
    render_launch_config(out, abi);
    let launch_args = if let Some(shared_bytes) = &abi.dynamic_shared_memory_bytes {
        format!("grid, block, {shared_bytes}")
    } else {
        "grid, block".to_string()
    };
    out.push_str(&format!(
        "    {}<<<{launch_args}>>>({});\n",
        program.entry.name,
        abi.kernel_arguments
            .iter()
            .map(String::as_str)
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
                out.push_str(&format!("{pad}if ({}) {{\n", abi.annotated_name(condition)));
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
            Stmt::Return(values) => {
                out.push_str(&format!("{pad}return;{}\n", return_comment(values, abi)));
            }
            Stmt::Barrier => out.push_str(&format!("{pad}__syncthreads();\n")),
            Stmt::Assign { lhs, rhs } => out.push_str(&format!(
                "{pad}{} = {};\n",
                abi.annotated_name(lhs),
                abi.annotated_name(rhs)
            )),
            Stmt::Call {
                function,
                inputs,
                outputs,
            } => {
                let function = sanitize_ident(function);
                let inputs = inputs
                    .iter()
                    .map(|name| abi.rename(name))
                    .collect::<Vec<_>>()
                    .join(", ");
                if outputs.is_empty() {
                    out.push_str(&format!("{pad}{function}({inputs});\n"));
                } else if outputs.len() == 1 {
                    out.push_str(&format!(
                        "{pad}{} = {function}({inputs});\n",
                        abi.rename(&outputs[0])
                    ));
                } else {
                    out.push_str(&format!(
                        "{pad}auto [{}] = {function}({inputs});\n",
                        outputs
                            .iter()
                            .map(|name| abi.rename(name))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
            Stmt::Primitive(primitive) => {
                let primitive = rename_primitive(primitive, abi);
                let comment = primitive_line_comment(&primitive, abi);
                for line in domain.lower_primitive_lines(&primitive, abi) {
                    out.push_str(&format!("{pad}{line}{comment}\n"));
                }
            }
            Stmt::Comment(comment) => out.push_str(&format!("{pad}// {comment}\n")),
        }
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn primitive_line_comment(primitive: &Primitive, abi: &CudaKernelAbi) -> String {
    let mut seen = std::collections::HashSet::new();
    let annotations = primitive
        .outputs
        .iter()
        .chain(&primitive.inputs)
        .filter(|name| seen.insert((*name).clone()))
        .filter_map(|name| {
            abi.source_name_annotation(name)
                .map(|annotation| format!("{name}: {}", sanitize_comment(annotation)))
        })
        .collect::<Vec<_>>();
    if annotations.is_empty() {
        String::new()
    } else {
        format!(" /* {} */", annotations.join(", "))
    }
}

fn return_comment(values: &[String], abi: &CudaKernelAbi) -> String {
    if values.is_empty() {
        return String::new();
    }
    let values = values
        .iter()
        .map(|value| match abi.source_name_annotation(value) {
            Some(annotation) => format!("{value}: {}", sanitize_comment(annotation)),
            None => value.clone(),
        })
        .collect::<Vec<_>>();
    format!(" /* returns {} */", values.join(", "))
}

fn sanitize_comment(comment: &str) -> String {
    comment.replace("*/", "* /")
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
