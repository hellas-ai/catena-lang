use super::super::{
    CodegenError, Lowerer, expr, invalid,
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
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let ty = single_output(outputs)?;
    let expression = match (op, xs.as_slice()) {
        (
            "stdlib.runtime.global_own_u32"
            | "stdlib.runtime.global_own_u64"
            | "stdlib.runtime.global_own_f32"
            | "stdlib.runtime.global_ref_u32"
            | "stdlib.runtime.global_ref_u64"
            | "stdlib.runtime.global_ref_f32",
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
        (
            "stdlib.runtime.global_u32_to_mem"
            | "stdlib.runtime.global_u64_to_mem"
            | "stdlib.runtime.global_f32_to_mem",
            [mem],
        ) => format!("catena_mem_own_t{{{mem}.data,{mem}.bytes}}"),
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    let v = l.emit(&ty, expression)?;
    results(l, outputs, vec![v])
}
