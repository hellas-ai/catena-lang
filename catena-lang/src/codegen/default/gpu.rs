//! Rendering is deliberately mechanical: arrow semantics live in `ops`.
use super::{ir::*, lower_types::CType};
use crate::runtime::GpuDialect;
use std::{collections::BTreeSet, fmt::Write};

pub(super) fn render(modules: &Modules, dialect: GpuDialect) -> String {
    let wmma_kernels = modules
        .kernels
        .values()
        .filter(|f| f.uses_wmma_mma())
        .map(|f| f.symbol.as_str())
        .collect::<BTreeSet<_>>();
    let mut out = super::prelude::render(dialect);
    if modules.bf16 {
        out.push_str(&super::bf16_support::render(
            dialect,
            modules.bf16_dot,
            modules.wmma,
        ));
    }
    if modules.nvidia_mma {
        out.push_str(super::nvidia_mma::SUPPORT);
    }
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
            if kernel && wmma_kernels.contains(f.symbol.as_str()) {
                out.push_str("template<bool CatenaWmmaEmulate>\n");
            }
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
            render_body(&mut out, &f.body, 1, dialect, &wmma_kernels);
            out.push_str("}\n");
        }
    }
    out
}

fn render_body(
    out: &mut String,
    body: &[Instruction],
    depth: usize,
    dialect: GpuDialect,
    wmma_kernels: &BTreeSet<&str>,
) {
    let indent = "    ".repeat(depth);
    for instruction in body {
        match instruction {
            Instruction::Let(v, e) => {
                writeln!(out, "{indent}{} {} = {e};", v.ty.c_name(), v.name).unwrap()
            }
            Instruction::NvidiaMma {
                result,
                a,
                b,
                accumulator,
            } => {
                writeln!(
                    out,
                    "{indent}{} {} = catena_mma_bf16_f32({},{},{});",
                    result.ty.c_name(),
                    result.name,
                    a.name,
                    b.name,
                    accumulator.name
                )
                .unwrap();
            }
            Instruction::WmmaMma {
                result,
                a,
                b,
                accumulator,
            } => {
                writeln!(
                    out,
                    "{indent}{} {} = catena_wmma_mma<CatenaWmmaEmulate>({},{},{});",
                    result.ty.c_name(),
                    result.name,
                    a.name,
                    b.name,
                    accumulator.name
                )
                .unwrap();
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
                if result.ty == CType::Bf16x2 {
                    writeln!(
                        out,
                        "{indent}catena_bf16x2_t {} = catena_bf16_load_pair(&{}.data[{}]);",
                        result.name, buffer.name, index.name
                    )
                    .unwrap();
                    continue;
                }
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
                if value.ty == CType::Bf16x2 {
                    writeln!(
                        out,
                        "{indent}catena_bf16_store_pair(&{}.data[{}], {});",
                        buffer.name, index.name, value.name
                    )
                    .unwrap();
                    continue;
                }
                writeln!(
                    out,
                    "{indent}{}.data[{}] = {};",
                    buffer.name, index.name, value.name
                )
                .unwrap();
            }
            Instruction::Free { buffer } => {
                writeln!(
                    out,
                    "{indent}catena_gpu_check({}({}.data));",
                    dialect.device_free_fn(),
                    buffer.name
                )
                .unwrap();
            }
            Instruction::If { condition, yes, no } => {
                writeln!(out, "{indent}if ({condition}) {{").unwrap();
                render_body(out, yes, depth + 1, dialect, wmma_kernels);
                writeln!(out, "{indent}}} else {{").unwrap();
                render_body(out, no, depth + 1, dialect, wmma_kernels);
                writeln!(out, "{indent}}}").unwrap();
            }
            Instruction::For {
                index,
                index_type,
                end,
                body,
            } => {
                writeln!(
                    out,
                    "{indent}for ({} {index}=0; {index}<{end}; ++{index}) {{",
                    index_type.c_name()
                )
                .unwrap();
                render_body(out, body, depth + 1, dialect, wmma_kernels);
                writeln!(out, "{indent}}}").unwrap();
            }
            Instruction::Launch {
                kernel,
                grid,
                shared_bytes,
                arguments,
            } => {
                let arguments = arguments.join(", ");
                if wmma_kernels.contains(kernel.as_str()) {
                    writeln!(out, "{indent}if (catena_wmma_emulation_enabled()) {{").unwrap();
                    writeln!(out,
                        "{indent}    {kernel}<true><<<{grid}.blocks,{grid}.threads,{shared_bytes}>>>({arguments});"
                    ).unwrap();
                    writeln!(out, "{indent}}} else {{").unwrap();
                    writeln!(out,
                        "{indent}    {kernel}<false><<<{grid}.blocks,{grid}.threads,{shared_bytes}>>>({arguments});"
                    ).unwrap();
                    writeln!(out, "{indent}}}").unwrap();
                } else {
                    writeln!(out,
                        "{indent}{kernel}<<<{grid}.blocks,{grid}.threads,{shared_bytes}>>>({arguments});"
                    ).unwrap();
                }
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
