//! Lower an explicit kernel function/environment pair into a GPU launch.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, children, runtime},
    values::*,
};

fn kernel_argument(l: &mut Lowerer<'_>, ty: &Obj, grid: &str) -> Result<Value, CodegenError> {
    if let Some(fields) = children(ty, "*") {
        return Ok(super::super::product(
            ty.clone(),
            fields
                .iter()
                .map(|ty| kernel_argument(l, ty, grid))
                .collect::<Result<_, _>>()?,
        ));
    }
    match runtime(ty)? {
        Some(CType::Block) => l.emit(
            ty,
            format!("exp_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}"),
        ),
        Some(CType::Thread) => {
            let block = format!("exp_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}");
            l.emit(
                ty,
                format!("exp_thread{{{block},{{threadIdx.x,threadIdx.y,threadIdx.z}}}}"),
            )
        }
        Some(_) => Err(invalid(
            "unsafe.launch",
            "kernel input must contain only block/thread data and proofs",
        )),
        None => l.erased(ty),
    }
}

pub(crate) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if l.place != Place::Host {
        return Err(invalid(op, "nested device launch is unsupported"));
    }
    let [grid, shared, environment, kernel, _, _] = args else {
        return Err(invalid(op, "invalid launch operands"));
    };
    if runtime(&grid.ty)? != Some(CType::Grid)
        || !matches!(runtime(&shared.ty)?, Some(CType::Layout(_)))
    {
        return Err(invalid(op, "expected a grid and shared-memory layout"));
    }
    let (domain, codomain) = super::super::function_parts(&kernel.ty)?;
    let grid_expression = expr(grid)?.to_owned();
    let shared_bytes = format!("{}.bytes", expr(shared)?);
    let grid_parameter = l.fresh("grid");
    let mut inputs = vec![Variable {
        name: grid_parameter.clone(),
        ty: CType::Grid,
    }];
    let mut arguments = vec![grid_expression.clone()];
    let mut captured = Vec::new();
    environment.clone().flatten(&mut captured);
    for value in &mut captured {
        if let Repr::Runtime(expression) = &mut value.repr {
            let name = l.fresh("capture");
            arguments.push(expression.clone());
            *expression = name.clone();
            inputs.push(Variable {
                name,
                ty: runtime(&value.ty)?.ok_or_else(|| invalid(op, "invalid capture type"))?,
            });
        }
    }
    let outer = std::mem::take(&mut l.body);
    l.place = Place::Device;
    let types = crate::pass::unpack_products::flatten_object(domain);
    let remaining = types
        .get(captured.len()..)
        .ok_or_else(|| invalid(op, "environment exceeds kernel domain"))?;
    for ty in remaining {
        captured.push(kernel_argument(l, ty, &grid_parameter)?);
    }
    let results = l.call_function(
        kernel,
        captured,
        &crate::pass::unpack_products::flatten_object(codomain),
    )?;
    for result in results {
        l.erased(&result.ty)?;
    }
    let body = std::mem::replace(&mut l.body, outer);
    l.place = Place::Host;
    let symbol = l.fresh("exp_kernel");
    l.modules.kernels.insert(
        symbol.clone(),
        Function {
            symbol: symbol.clone(),
            inputs,
            outputs: vec![],
            body,
        },
    );
    l.body.push(Instruction::Launch {
        kernel: symbol,
        grid: grid_expression,
        shared_bytes,
        arguments,
    });
    outputs.iter().map(|ty| l.erased(ty)).collect()
}
