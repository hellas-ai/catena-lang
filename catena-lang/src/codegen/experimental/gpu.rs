//! Rendering is deliberately mechanical: arrow semantics live in `ops`.
use super::ir::*;
use crate::runtime::GpuDialect;
use std::fmt::Write;

pub(super) fn render(modules: &Modules, dialect: GpuDialect) -> String {
    let mut out = super::prelude::render(dialect);
    for (kernel, functions) in [(true, &modules.kernels), (false, &modules.functions)] {
        for f in functions.values() {
            let mut parameters = f
                .inputs
                .iter()
                .map(|v| format!("{} {}", v.ty.c_name(), v.name))
                .collect::<Vec<_>>();
            parameters.extend(
                f.outputs
                    .iter()
                    .map(|v| format!("{}* {}", v.ty.c_name(), v.name)),
            );
            writeln!(
                out,
                "{} void {}({}) {{",
                if kernel {
                    "__global__"
                } else {
                    "extern \"C\" __host__"
                },
                f.symbol,
                parameters.join(", ")
            )
            .unwrap();
            if kernel {
                out.push_str(
                    "    extern __shared__ __align__(16) unsigned char catena_shared[];\n",
                );
            }
            render_body(&mut out, &f.body, 1, dialect);
            out.push_str("}\n");
        }
    }
    out
}

fn render_body(out: &mut String, body: &[Instruction], depth: usize, dialect: GpuDialect) {
    let indent = "    ".repeat(depth);
    for instruction in body {
        match instruction {
            Instruction::Let(v, e) => {
                writeln!(out, "{indent}{} {} = {e};", v.ty.c_name(), v.name).unwrap()
            }
            Instruction::Assign(v, e) => writeln!(out, "{indent}{v} = {e};").unwrap(),
            Instruction::Sync => writeln!(out, "{indent}__syncthreads();").unwrap(),
            Instruction::Assert { condition } => {
                writeln!(out, "{indent}catena_assert({condition});").unwrap();
            }
            Instruction::Load {
                result,
                buffer,
                index,
            } => {
                writeln!(
                    out,
                    "{indent}{} {} = {}.data[{}];",
                    result.ty.c_name(),
                    result.name,
                    buffer.name,
                    index.name
                )
                .unwrap();
            }
            Instruction::Store {
                buffer,
                index,
                value,
            } => {
                writeln!(
                    out,
                    "{indent}{}.data[{}] = {};",
                    buffer.name, index.name, value.name
                )
                .unwrap();
            }
            Instruction::If { condition, yes, no } => {
                writeln!(out, "{indent}if ({condition}) {{").unwrap();
                render_body(out, yes, depth + 1, dialect);
                writeln!(out, "{indent}}} else {{").unwrap();
                render_body(out, no, depth + 1, dialect);
                writeln!(out, "{indent}}}").unwrap();
            }
            Instruction::For { index, end, body } => {
                writeln!(
                    out,
                    "{indent}for (uint32_t {index}=0; {index}<{end}; ++{index}) {{"
                )
                .unwrap();
                render_body(out, body, depth + 1, dialect);
                writeln!(out, "{indent}}}").unwrap();
            }
            Instruction::Launch {
                kernel,
                grid,
                shared_bytes,
                arguments,
            } => {
                writeln!(
                    out,
                    "{indent}{kernel}<<<{grid}.blocks,{grid}.threads,{shared_bytes}>>>({});",
                    arguments.join(", ")
                )
                .unwrap();
                let last_error = match dialect {
                    GpuDialect::Hip => "hipGetLastError",
                    GpuDialect::Cuda => "cudaGetLastError",
                };
                writeln!(out, "{indent}catena_gpu_check({last_error}());").unwrap();
                writeln!(
                    out,
                    "{indent}catena_gpu_check({}());",
                    dialect.synchronize_fn()
                )
                .unwrap();
            }
        }
    }
}
