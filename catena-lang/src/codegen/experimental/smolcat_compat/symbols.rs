use super::super::{
    CodegenError, Lowerer, invalid,
    lower_types::{children, runtime},
    values::{Obj, Value},
};

pub(super) fn lower(
    l: &Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if op == "smolcat.callback.symbol" {
        for value in args {
            l.erased(&value.ty)?;
        }
        return outputs.iter().map(|ty| l.erased(ty)).collect();
    }
    let [ty] = outputs else {
        return Err(invalid(op, "value naming requires one output"));
    };
    let value = match (op, args) {
        ("smolcat.value.named", [value]) => value,
        ("smolcat.value.named_at", [value, witness]) => {
            l.erased(&witness.ty)?;
            let Some([identity, _]) = children(ty, ":") else {
                return Err(invalid(op, "named_at must return a named value"));
            };
            if identity != &witness.ty {
                return Err(invalid(op, "named_at must preserve the supplied identity"));
            }
            value
        }
        _ => return Err(invalid(op, "invalid value naming operands")),
    };
    if runtime(&value.ty)? != runtime(ty)? {
        return Err(invalid(
            op,
            "value naming must preserve runtime representation",
        ));
    }
    let mut result = value.clone();
    result.ty = ty.clone();
    Ok(vec![result])
}
