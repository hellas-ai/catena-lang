//! Lower an explicit kernel function/environment pair into a GPU launch.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, children, runtime},
    values::*,
};

/// Runtime expressions are launch arguments, not kernel specializations. Keep
/// full Hex types and compile-time function captures in the key: equal C++
/// representations alone do not imply equal callback behavior.
#[derive(PartialEq, Eq)]
pub(in crate::codegen::experimental) struct KernelKey {
    linear: bool,
    function: hexpr::Operation,
    function_type: Obj,
    captures: Vec<(Obj, CaptureKind)>,
}

#[derive(PartialEq, Eq)]
enum CaptureKind {
    Runtime,
    Erased,
    Function(hexpr::Operation),
}

fn key(linear: bool, kernel: &Value, captured: &[Value]) -> Option<KernelKey> {
    let Repr::Function(function) = &kernel.repr else {
        return None;
    };
    let captures = captured
        .iter()
        .map(|value| {
            let function = match &value.repr {
                Repr::Runtime(_) => CaptureKind::Runtime,
                Repr::Erased => CaptureKind::Erased,
                Repr::Function(function) => CaptureKind::Function(function.clone()),
                Repr::Product(_) => return None,
            };
            Some((value.ty.clone(), function))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(KernelKey {
        linear,
        function: function.clone(),
        function_type: kernel.ty.clone(),
        captures,
    })
}

fn cached(l: &Lowerer<'_>, key: &Option<KernelKey>) -> Option<String> {
    let key = key.as_ref()?;
    l.kernel_cache
        .iter()
        .find(|(previous, _)| previous == key)
        .map(|(_, symbol)| symbol.clone())
}

fn kernel_argument(
    l: &mut Lowerer<'_>,
    op: &str,
    ty: &Obj,
    grid: &str,
) -> Result<Value, CodegenError> {
    if let Some(fields) = children(ty, "*") {
        return Ok(super::super::product(
            ty.clone(),
            fields
                .iter()
                .map(|ty| kernel_argument(l, op, ty, grid))
                .collect::<Result<_, _>>()?,
        ));
    }
    match runtime(ty)? {
        Some(CType::Block) => l.emit(
            ty,
            format!("catena_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}"),
        ),
        Some(CType::Thread) => {
            let block = format!("catena_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}");
            l.emit(
                ty,
                format!("catena_thread{{{block},{{threadIdx.x,threadIdx.y,threadIdx.z}}}}"),
            )
        }
        Some(_) => Err(invalid(
            op,
            "kernel input must contain only block/thread data and proofs",
        )),
        None => l.erased(ty),
    }
}

pub(in crate::codegen::experimental) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if l.place != Place::Host {
        return Err(invalid(op, "nested device launch is unsupported"));
    }
    // Closure conversion expands the kernel into environment + function operands.
    let (grid, shared, environment, kernel) = match (op, args) {
        ("unsafe.launch", [grid, environment, kernel]) => (grid, None, environment, kernel),
        ("unsafe.launch_shared", [grid, shared, environment, kernel]) => {
            (grid, Some(shared), environment, kernel)
        }
        _ => return Err(invalid(op, "invalid launch operands")),
    };
    if runtime(&grid.ty)? != Some(CType::Grid) {
        return Err(invalid(op, "expected a grid"));
    }
    let shared_bytes = if let Some(shared) = shared {
        if !matches!(runtime(&shared.ty)?, Some(CType::Layout(_))) {
            return Err(invalid(op, "expected shared memory"));
        }
        format!("{}.bytes", expr(shared)?)
    } else {
        "0".into()
    };
    let (domain, codomain) = super::super::function_parts(&kernel.ty)?;
    let grid_expression = expr(grid)?.to_owned();
    let grid_parameter = l.fresh("grid");
    let mut inputs = vec![Variable {
        name: grid_parameter.clone(),
        ty: CType::Grid,
    }];
    let mut arguments = vec![grid_expression.clone()];
    let mut captured = Vec::new();
    environment.clone().flatten(&mut captured);
    for value in &captured {
        if matches!(value.repr, Repr::Runtime(_)) && runtime(&value.ty)? == Some(CType::MemRefs) {
            return Err(invalid(
                op,
                "host memory-reference tables cannot be captured by a GPU kernel",
            ));
        }
    }
    let specialization = key(false, kernel, &captured);
    if let Some(symbol) = cached(l, &specialization) {
        for value in &captured {
            if let Repr::Runtime(expression) = &value.repr {
                arguments.push(expression.clone());
            }
        }
        l.body.push(Instruction::Launch {
            kernel: symbol,
            grid: grid_expression,
            shared_bytes,
            arguments,
        });
        return outputs.iter().map(|ty| l.erased(ty)).collect();
    }
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
        captured.push(kernel_argument(l, op, ty, &grid_parameter)?);
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
    let symbol = l.fresh("catena_kernel");
    l.modules.kernels.insert(
        symbol.clone(),
        Function {
            symbol: symbol.clone(),
            inputs,
            outputs: vec![],
            body,
        },
    );
    if let Some(specialization) = specialization {
        l.kernel_cache.push((specialization, symbol.clone()));
    }
    l.body.push(Instruction::Launch {
        kernel: symbol,
        grid: grid_expression,
        shared_bytes,
        arguments,
    });
    outputs.iter().map(|ty| l.erased(ty)).collect()
}

/// A guarded 1-D launch for a plain u32 index callback. Captures are still
/// explicit closure-converted environment operands, as for `unsafe.launch`.
pub(in crate::codegen::experimental) fn linear(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    use super::super::lower_types::node;
    if l.place != Place::Host {
        return Err(invalid(op, "launch requires host execution"));
    }
    let [count, environment, kernel] = args else {
        return Err(invalid(op, "invalid launch operands"));
    };
    if runtime(&count.ty)? != Some(CType::U32) {
        return Err(invalid(op, "count must be u32"));
    }
    let grid_ty = node(
        "val",
        vec![node(
            "stdlib.gpu.geometry.type.Grid",
            vec![node("1", vec![]), node("1", vec![])],
        )],
    );
    let count_expr = expr(count)?.to_owned();
    let grid = l.emit(
        &grid_ty,
        format!("catena_make_grid({count_expr}/256+({count_expr}%256 != 0),1,256,1)"),
    )?;
    let grid_param = l.fresh("grid");
    let count_param = l.fresh("count");
    let mut inputs = vec![
        Variable {
            name: grid_param,
            ty: CType::Grid,
        },
        Variable {
            name: count_param.clone(),
            ty: CType::U32,
        },
    ];
    let mut arguments = vec![expr(&grid)?.into(), count_expr];
    let mut captured = vec![];
    environment.clone().flatten(&mut captured);
    for value in &captured {
        if matches!(value.repr, Repr::Runtime(_)) && runtime(&value.ty)? == Some(CType::MemRefs) {
            return Err(invalid(
                op,
                "host memory-reference tables cannot be captured by a GPU kernel",
            ));
        }
    }
    let specialization = key(true, kernel, &captured);
    if let Some(symbol) = cached(l, &specialization) {
        for value in &captured {
            if let Repr::Runtime(expression) = &value.repr {
                arguments.push(expression.clone());
            }
        }
        l.body.push(Instruction::Launch {
            kernel: symbol,
            grid: expr(&grid)?.into(),
            shared_bytes: "0".into(),
            arguments,
        });
        return outputs.iter().map(|ty| l.erased(ty)).collect();
    }
    for value in &mut captured {
        if let Repr::Runtime(expression) = &mut value.repr {
            let name = l.fresh("capture");
            arguments.push(expression.clone());
            *expression = name.clone();
            inputs.push(Variable {
                name,
                ty: runtime(&value.ty)?.ok_or_else(|| invalid(op, "invalid capture"))?,
            });
        }
    }
    let outer = std::mem::take(&mut l.body);
    l.place = Place::Device;
    let index_ty = node("val", vec![node("u32", vec![])]);
    let index = l.emit(&index_ty, "uint32_t(blockIdx.x)*256+uint32_t(threadIdx.x)")?;
    captured.push(index.clone());
    let (_, codomain) = super::super::function_parts(&kernel.ty)?;
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
    // The index declaration precedes the guard; all callback work is predicated.
    let mut body = body.into_iter();
    let index_decl = body.next().ok_or_else(|| invalid(op, "missing index"))?;
    let symbol = l.fresh("catena_kernel");
    l.modules.kernels.insert(
        symbol.clone(),
        Function {
            symbol: symbol.clone(),
            inputs,
            outputs: vec![],
            body: vec![
                index_decl,
                Instruction::If {
                    condition: format!("{} < {count_param}", expr(&index)?),
                    yes: body.collect(),
                    no: vec![],
                },
            ],
        },
    );
    if let Some(specialization) = specialization {
        l.kernel_cache.push((specialization, symbol.clone()));
    }
    l.body.push(Instruction::Launch {
        kernel: symbol,
        grid: expr(&grid)?.into(),
        shared_bytes: "0".into(),
        arguments,
    });
    outputs.iter().map(|ty| l.erased(ty)).collect()
}
