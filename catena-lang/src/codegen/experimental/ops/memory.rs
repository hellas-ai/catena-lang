use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, children, node, runtime},
    values::*,
};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let name = op.trim_start_matches("stdlib.gpu.memory.");
    if name == "allocate" && l.place != Place::Host {
        return Err(invalid(op, "allocation requires host execution"));
    }
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    if matches!(
        name,
        "memoryelement.u32" | "memoryelement.u64" | "memoryelement.f32"
    ) {
        return outputs.iter().map(|ty| l.erased(ty)).collect();
    }
    if name == "free" {
        if l.place != Place::Host || values.len() != 1 {
            return Err(invalid(op, "free requires one global buffer on the host"));
        }
        let buffer = operand(&values[0])?;
        if !matches!(buffer.ty, CType::Global(_)) {
            return Err(invalid(op, "free requires a global buffer"));
        }
        l.body.push(Instruction::Free { buffer });
        return results(l, outputs, vec![]);
    }
    if matches!(
        name,
        "mem_cast_own_f32" | "mem_cast_own_u64" | "mem_cast_ref_f32" | "mem_cast_ref_u64"
    ) {
        let [memory] = values.as_slice() else {
            return Err(invalid(op, "mem_cast requires one memory operand"));
        };
        let element = if name.ends_with("f32") {
            "float"
        } else {
            "uint64_t"
        };
        let count_ty = node("val", vec![node("u64", vec![])]);
        let count = l.emit(
            &count_ty,
            format!("catena_mem_count<{element}>({})", expr(memory)?),
        )?;
        fn global_type(outputs: &[Obj]) -> Option<Obj> {
            for ty in outputs {
                if let Some(fields) = children(ty, "*") {
                    if let Some(found) = global_type(fields) {
                        return Some(found);
                    }
                } else if matches!(runtime(ty), Ok(Some(CType::Global(_)))) {
                    return Some(ty.clone());
                }
            }
            None
        }
        let ty = global_type(outputs).ok_or_else(|| invalid(op, "missing global result"))?;
        let global = l.emit(
            &ty,
            format!(
                "catena_global_from_mem<{element}>({},{})",
                expr(memory)?,
                expr(&count)?
            ),
        )?;
        return results(l, outputs, vec![count, global]);
    }
    if matches!(
        name,
        "global_read"
            | "global_read_unsafe"
            | "global_write"
            | "shared_read"
            | "shared_write"
            | "shared_slot"
    ) && l.place != Place::Device
    {
        return Err(invalid(op, "requires device execution"));
    }
    if matches!(name, "global_write" | "shared_write") {
        let [memory, index, value] = values.as_slice() else {
            return Err(invalid(op, "invalid write operands"));
        };
        let buffer = operand(memory)?;
        let index = operand(index)?;
        l.body.push(Instruction::Assert {
            condition: format!("{} < {}.count", index.name, buffer.name),
        });
        l.body.push(Instruction::Store {
            buffer,
            index,
            value: operand(value)?,
        });
        return results(l, outputs, vec![]);
    }
    let ty = single_output(outputs)?;
    if matches!(name, "global_read" | "global_read_unsafe" | "shared_read") {
        let (memory, index) = match (name, values.as_slice()) {
            ("global_read", [_, memory, index])
            | ("global_read_unsafe" | "shared_read", [memory, index]) => (memory, index),
            _ => return Err(invalid(op, "invalid read operands")),
        };
        let buffer = operand(memory)?;
        let index = operand(index)?;
        let name = l.fresh("v");
        let result = Variable {
            name: name.clone(),
            ty: runtime(&ty)?.ok_or_else(|| invalid(op, "load requires a runtime result"))?,
        };
        l.body.push(Instruction::Assert {
            condition: format!("{} < {}.count", index.name, buffer.name),
        });
        l.body.push(Instruction::Load {
            result,
            buffer,
            index,
        });
        return results(
            l,
            outputs,
            vec![Value {
                ty,
                repr: Repr::Runtime(name),
            }],
        );
    }
    let expression = match (name, xs.as_slice()) {
        ("allocate", [bytes]) if runtime(&ty)? == Some(CType::MemOwn) => {
            format!("catena_allocate({bytes})")
        }
        (
            "global_own_u32" | "global_own_u64" | "global_own_f32" | "global_ref_u32"
            | "global_ref_u64" | "global_ref_f32",
            [mem, count],
        ) => {
            let Some(CType::Global(element)) = runtime(&ty)? else {
                return Err(invalid(op, "expected Global result"));
            };
            format!(
                "catena_global_from_mem<{}>({mem},{count})",
                element.c_name()
            )
        }
        ("global_u32_to_mem" | "global_u64_to_mem" | "global_f32_to_mem", [mem]) => {
            format!("catena_mem_own_t{{{mem}.data,{mem}.bytes}}")
        }
        ("global_cast_equal_own" | "global_cast_equal_ref", [mem, count]) => {
            l.body.push(Instruction::Assert {
                condition: format!("{mem}.count == {count}"),
            });
            (*mem).into()
        }
        ("ix" | "slot_name", [v]) => (*v).into(),
        ("size", [v]) => format!("{v}.count"),
        ("empty_shared_layout", []) => "{}".into(),
        ("add_shared_slot", [name, size, count, layout]) => {
            format!("catena_add_slot({layout},{name},{size},{count})")
        }
        ("reserve_shared", [_, layout]) => (*layout).into(),
        ("shared_slot", [name, layout, _, count]) => {
            let Some(CType::Shared(element)) = runtime(&ty)? else {
                return Err(invalid(op, "expected shared view"));
            };
            format!(
                "catena_slot<{}>(catena_shared,{layout},{name},{count})",
                element.c_name()
            )
        }
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    let value = l.emit(&ty, expression)?;
    results(l, outputs, vec![value])
}

fn operand(value: &Value) -> Result<Variable, CodegenError> {
    Ok(Variable {
        name: expr(value)?.into(),
        ty: runtime(&value.ty)?
            .ok_or_else(|| invalid("memory access", "expected runtime operand"))?,
    })
}
