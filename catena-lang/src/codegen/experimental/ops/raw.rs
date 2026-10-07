//! Checked plain-value adapters for programs without symbolic value identities.
//! These are generic runtime operations, not model-specific implementations.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, runtime},
    values::*,
};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if matches!(
        op,
        "stdlib.runtime.raw.mem_refs_empty"
            | "stdlib.runtime.raw.mem_refs_push"
            | "stdlib.runtime.raw.mem_ref_at"
    ) {
        if l.place != Place::Host {
            return Err(invalid(
                op,
                "memory-reference tables require host execution",
            ));
        }
        let ty = single_output(outputs)?;
        let expression = if op == "stdlib.runtime.raw.mem_refs_empty" {
            if !runtime_args(args).is_empty() || runtime(&ty)? != Some(CType::MemRefs) {
                return Err(invalid(op, "expected no runtime inputs and a table result"));
            }
            "catena_mem_refs{}".into()
        } else if op == "stdlib.runtime.raw.mem_refs_push" {
            let [table, reference] = args else {
                return Err(invalid(op, "expected a table and borrowed buffer"));
            };
            if runtime(&table.ty)? != Some(CType::MemRefs)
                || runtime(&reference.ty)? != Some(CType::MemRef)
                || runtime(&ty)? != Some(CType::MemRefs)
            {
                return Err(invalid(
                    op,
                    "only borrowed buffers can be appended to a table",
                ));
            }
            format!(
                "catena_mem_refs_push({},{})",
                expr(table)?,
                expr(reference)?
            )
        } else {
            let [table, index] = args else {
                return Err(invalid(op, "expected a table and u32 index"));
            };
            if runtime(&table.ty)? != Some(CType::MemRefs)
                || runtime(&index.ty)? != Some(CType::U32)
                || runtime(&ty)? != Some(CType::MemRef)
            {
                return Err(invalid(op, "invalid table, index, or result type"));
            }
            let table = expr(table)?;
            let index = expr(index)?;
            l.body.push(Instruction::Assert {
                condition: format!("{index} < {table}.size()"),
            });
            format!("{table}[{index}]")
        };
        let result = l.emit(&ty, expression)?;
        return results(l, outputs, vec![result]);
    }
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let types = values
        .iter()
        .map(|v| runtime(&v.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let name = op.rsplit('.').next().unwrap();
    if op.starts_with("stdlib.numeric.raw.") {
        let ty = single_output(outputs)?;
        let out = runtime(&ty)?.unwrap();
        let expression = match (name, xs.as_slice(), types.as_slice(), &out) {
            (
                "add" | "sub" | "mul" | "div" | "rem" | "less" | "less_equal" | "equal" | "max",
                [a, b],
                [Some(t), Some(u)],
                _,
            ) if t == u && t.numeric() => {
                let comparison = matches!(name, "less" | "less_equal" | "equal");
                if out != if comparison { CType::Bool } else { t.clone() } {
                    return Err(invalid(op, "numeric result type mismatch"));
                }
                if name == "rem" && !t.integral() {
                    return Err(invalid(op, "remainder requires integers"));
                }
                if matches!(name, "div" | "rem") && t.integral() {
                    l.body.push(Instruction::Assert {
                        condition: format!("{b} != 0"),
                    });
                }
                if name == "max" {
                    format!("({a} > {b} ? {a} : {b})")
                } else {
                    let operator = match name {
                        "add" => "+",
                        "sub" => "-",
                        "mul" => "*",
                        "div" => "/",
                        "rem" => "%",
                        "less" => "<",
                        "less_equal" => "<=",
                        "equal" => "==",
                        _ => unreachable!(),
                    };
                    if *t == CType::U16 {
                        format!("(uint32_t({a}) {operator} uint32_t({b}))")
                    } else {
                        format!("({a} {operator} {b})")
                    }
                }
            }
            ("select", [c, a, b], [Some(CType::Bool), Some(t), Some(u)], _)
                if t == u && *t == out =>
            {
                format!("({c} ? {a} : {b})")
            }
            ("negate", [x], [Some(CType::F32)], CType::F32) => format!("-{x}"),
            ("to_f32", [x], [Some(CType::U32) | Some(CType::U64)], CType::F32) => {
                format!("float({x})")
            }
            ("to_u32", [x], [Some(CType::U64)], CType::U32) => {
                l.body.push(Instruction::Assert {
                    condition: format!("{x} <= UINT32_MAX"),
                });
                format!("uint32_t({x})")
            }
            ("to_u64", [x], [Some(CType::U32)], CType::U64) => format!("uint64_t({x})"),
            ("round_u32", [x], [Some(CType::F32)], CType::U32) => format!("catena_round_u32({x})"),
            ("bits_u32", [x], [Some(CType::F32)], CType::U32) => {
                format!("catena_f32_bitcast_u32({x})")
            }
            ("shr", [x, n], [Some(CType::U32), Some(CType::U32)], CType::U32) => {
                l.body.push(Instruction::Assert {
                    condition: format!("{n} < 32"),
                });
                format!("({x} >> {n})")
            }
            _ => return Err(invalid(op, "unsupported numeric operands")),
        };
        let v = l.emit(&ty, expression)?;
        return results(l, outputs, vec![v]);
    }
    let is_mem = |ty: &Option<CType>| matches!(ty, Some(CType::MemOwn | CType::MemRef));
    match (name, xs.as_slice(), types.as_slice()) {
        ("borrow", [mem], [Some(CType::MemOwn)]) => {
            if outputs.len() != 2
                || runtime(&outputs[0])? != Some(CType::MemOwn)
                || runtime(&outputs[1])? != Some(CType::MemRef)
            {
                return Err(invalid(
                    op,
                    "borrow must retain ownership and return a reference",
                ));
            }
            let reference = l.emit(
                &outputs[1],
                format!("catena_mem_ref_t{{{mem}.data,{mem}.len}}"),
            )?;
            return results(l, outputs, vec![values[0].clone(), reference]);
        }
        ("free", [mem], [Some(CType::MemOwn)]) => {
            if l.place != Place::Host {
                return Err(invalid(op, "free requires host execution"));
            }
            let name = l.fresh("freed");
            l.body.push(Instruction::Let(
                Variable {
                    name,
                    ty: CType::U32,
                },
                format!("catena_free({mem})"),
            ));
            return outputs.iter().map(|t| l.erased(t)).collect();
        }
        (
            "write_f32",
            [mem, index, value],
            [Some(CType::MemOwn), Some(CType::U32), Some(CType::F32)],
        ) => {
            if l.place != Place::Device {
                return Err(invalid(op, "write requires device execution"));
            }
            let buffer = l.fresh("buffer");
            let var = Variable {
                name: buffer.clone(),
                ty: CType::Global(Box::new(CType::F32)),
            };
            l.body.push(Instruction::Let(
                var.clone(),
                format!("catena_global_bytes<float>({mem})"),
            ));
            l.body.push(Instruction::Assert {
                condition: format!("{index} < {buffer}.count"),
            });
            l.body.push(Instruction::Store {
                buffer: var,
                index: Variable {
                    name: (*index).into(),
                    ty: CType::U32,
                },
                value: Variable {
                    name: (*value).into(),
                    ty: CType::F32,
                },
            });
            return outputs.iter().map(|t| l.erased(t)).collect();
        }
        ("read_f32" | "read_u64", [mem, index], [m, Some(CType::U32)]) if is_mem(m) => {
            if l.place != Place::Device {
                return Err(invalid(op, "read requires device execution"));
            }
            let ty = single_output(outputs)?;
            let element = if name == "read_f32" {
                CType::F32
            } else {
                CType::U64
            };
            if runtime(&ty)? != Some(element.clone()) {
                return Err(invalid(op, "read result type mismatch"));
            }
            let buffer = l.fresh("buffer");
            let var = Variable {
                name: buffer.clone(),
                ty: CType::Global(Box::new(element.clone())),
            };
            l.body.push(Instruction::Let(
                var.clone(),
                format!("catena_global_bytes<{}>({mem})", element.c_name()),
            ));
            l.body.push(Instruction::Assert {
                condition: format!("{index} < {buffer}.count"),
            });
            let result = Variable {
                name: l.fresh("load"),
                ty: element,
            };
            l.body.push(Instruction::Load {
                result: result.clone(),
                buffer: var,
                index: Variable {
                    name: (*index).into(),
                    ty: CType::U32,
                },
            });
            return results(
                l,
                outputs,
                vec![Value {
                    ty,
                    repr: Repr::Runtime(result.name),
                }],
            );
        }
        _ => {}
    }
    let ty = single_output(outputs)?;
    let out = runtime(&ty)?;
    let expression = match (name, xs.as_slice(), types.as_slice(), out) {
        ("alloc_f32", [count], [Some(CType::U32)], Some(CType::MemOwn)) => {
            if l.place != Place::Host {
                return Err(invalid(op, "allocation requires host execution"));
            }
            format!("catena_alloc_f32({count})")
        }
        ("bytes", [mem], [m], Some(CType::U64)) if is_mem(m) => format!("{mem}.len"),
        _ => return Err(invalid(op, "unsupported memory operands")),
    };
    let value = l.emit(&ty, expression)?;
    results(l, outputs, vec![value])
}
