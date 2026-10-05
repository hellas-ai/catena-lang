use super::super::{CodegenError, Lowerer, expr, values::*};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let values = runtime_args(args);
    let xs = values.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let expression = match (op.trim_start_matches("stdlib.gpu.geometry."), xs.as_slice()) {
        ("make_grid", [bx, by, tx, ty]) => format!("exp_make_grid({bx},{by},{tx},{ty})"),
        ("block_index" | "in_block_index", [v]) => format!("{v}.index"),
        ("in_grid_index", [v]) => format!("exp_grid_index({v})"),
        ("x", [v]) => format!("{v}.x"),
        ("y", [v]) => format!("{v}.y"),
        ("z", [v]) => format!("{v}.z"),
        ("row_major", [r, c, rows, cols]) => format!("exp_row_major({r},{c},{rows},{cols})"),
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    let value = l.emit(&single_output(outputs)?, expression)?;
    results(l, outputs, vec![value])
}
