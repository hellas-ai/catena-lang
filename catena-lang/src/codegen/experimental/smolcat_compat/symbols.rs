use super::super::{
    CodegenError, Lowerer, invalid,
    lower_types::runtime,
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
    let ([value], [ty]) = (args, outputs) else {
        return Err(invalid(op, "expected one input and output"));
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
