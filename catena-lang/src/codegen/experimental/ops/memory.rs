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
    let name = op.trim_start_matches("stdlib.gpu.memory.");
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    if matches!(
        name,
        "global_read" | "global_write" | "shared_read" | "shared_write" | "shared_slot"
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
    if matches!(name, "global_read" | "shared_read") {
        let (memory, index) = match (name, values.as_slice()) {
            ("global_read", [_, memory, index]) | ("shared_read", [memory, index]) => {
                (memory, index)
            }
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
        ("ix" | "slot_name", [v]) => (*v).into(),
        ("size", [v]) => format!("{v}.count"),
        ("empty_shared_layout", []) => "{}".into(),
        ("add_shared_slot", [name, size, count, layout]) => {
            format!("exp_add_slot({layout},{name},{size},{count})")
        }
        ("reserve_shared", [_, layout]) => (*layout).into(),
        ("shared_slot", [name, layout, _, count]) => {
            let Some(CType::Shared(element)) = runtime(&ty)? else {
                return Err(invalid(op, "expected shared view"));
            };
            format!(
                "exp_slot<{}>(exp_shared,{layout},{name},{count})",
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
