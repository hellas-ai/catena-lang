//! Structural GPU constraints checked before rendering. This does not establish
//! that barriers are reached uniformly by every thread in a block.
use super::{CodegenError, ir::*, lower_types::CType};

pub(super) fn validate(modules: &Modules) -> Result<(), CodegenError> {
    for (place, functions) in [
        (Place::Host, &modules.functions),
        (Place::Device, &modules.kernels),
    ] {
        for function in functions.values() {
            if place == Place::Device && !function.outputs.is_empty() {
                return Err(error(
                    &function.symbol,
                    "kernels cannot have output parameters",
                ));
            }
            validate_body(modules, &function.symbol, place, &function.body)?;
        }
    }
    Ok(())
}

fn error(function: &str, reason: &str) -> CodegenError {
    CodegenError::InvalidIr {
        function: function.into(),
        reason: reason.into(),
    }
}

fn validate_body(
    modules: &Modules,
    function: &str,
    place: Place,
    body: &[Instruction],
) -> Result<(), CodegenError> {
    for instruction in body {
        match instruction {
            Instruction::Sync => {
                if place != Place::Device {
                    return Err(error(
                        function,
                        "block synchronization requires device execution",
                    ));
                }
            }
            Instruction::Load {
                result,
                buffer,
                index,
            } => {
                validate_memory(function, place, buffer, index, result)?;
            }
            Instruction::Store {
                buffer,
                index,
                value,
            } => {
                validate_memory(function, place, buffer, index, value)?;
            }
            Instruction::If { yes, no, .. } => {
                validate_body(modules, function, place, yes)?;
                validate_body(modules, function, place, no)?;
            }
            Instruction::For { body, .. } => validate_body(modules, function, place, body)?,
            Instruction::Launch {
                kernel, arguments, ..
            } => {
                if place != Place::Host {
                    return Err(error(function, "kernel launches require host execution"));
                }
                let target = modules
                    .kernels
                    .get(kernel)
                    .ok_or_else(|| error(function, "launch references an unknown kernel"))?;
                if target.inputs.len() != arguments.len() {
                    return Err(error(
                        function,
                        "launch argument count does not match its kernel",
                    ));
                }
            }
            Instruction::Let(..) | Instruction::Assign(..) | Instruction::Assert { .. } => {}
        }
    }
    Ok(())
}

fn validate_memory(
    function: &str,
    place: Place,
    buffer: &Variable,
    index: &Variable,
    value: &Variable,
) -> Result<(), CodegenError> {
    if place != Place::Device {
        return Err(error(
            function,
            "GPU memory accesses require device execution",
        ));
    }
    let (CType::Global(element) | CType::Shared(element)) = &buffer.ty else {
        return Err(error(
            function,
            "memory access requires a global or shared buffer",
        ));
    };
    if index.ty != CType::U32 {
        return Err(error(function, "memory index must be u32"));
    }
    if element.as_ref() != &value.ty {
        return Err(error(
            function,
            "memory value type does not match the buffer element type",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::GpuDialect;

    fn variable(name: &str, ty: CType) -> Variable {
        Variable {
            name: name.into(),
            ty,
        }
    }

    fn module(place: Place, body: Vec<Instruction>) -> Modules {
        let mut modules = Modules::default();
        let functions = match place {
            Place::Host => &mut modules.functions,
            Place::Device => &mut modules.kernels,
        };
        functions.insert(
            "entry".into(),
            Function {
                symbol: "entry".into(),
                inputs: vec![],
                outputs: vec![],
                body,
            },
        );
        modules
    }

    fn nested(body: Vec<Instruction>) -> Vec<Instruction> {
        vec![Instruction::For {
            index: "i".into(),
            end: "n".into(),
            body: vec![Instruction::If {
                condition: "condition".into(),
                yes: vec![],
                no: body,
            }],
        }]
    }

    #[test]
    fn synchronization_placement_is_checked_inside_control_flow() {
        let body = nested(vec![Instruction::Sync]);
        let host = module(Place::Host, body.clone());
        assert!(validate(&host).is_err());
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            assert!(super::super::render_runtime_module(&host, dialect).is_err());
            let device =
                super::super::render_runtime_module(&module(Place::Device, body.clone()), dialect)
                    .unwrap();
            assert_eq!(device.source.matches("__syncthreads();").count(), 1);
        }
    }

    #[test]
    fn memory_accesses_check_place_address_space_index_and_element() {
        for ty in [
            CType::Global(Box::new(CType::U32)),
            CType::Shared(Box::new(CType::U32)),
        ] {
            let buffer = variable("buffer", ty);
            let index = variable("index", CType::U32);
            let value = variable("value", CType::U32);
            let load = Instruction::Load {
                result: value.clone(),
                buffer: buffer.clone(),
                index: index.clone(),
            };
            let store = Instruction::Store {
                buffer: buffer.clone(),
                index: index.clone(),
                value: value.clone(),
            };
            let body = nested(vec![load, store]);
            assert!(validate(&module(Place::Device, body.clone())).is_ok());
            assert!(validate(&module(Place::Host, body)).is_err());
            for (buffer, index, value) in [
                (
                    variable("bad_buffer", CType::U32),
                    index.clone(),
                    value.clone(),
                ),
                (
                    buffer.clone(),
                    variable("bad_index", CType::F32),
                    value.clone(),
                ),
                (
                    buffer.clone(),
                    index.clone(),
                    variable("wrong_element", CType::F32),
                ),
            ] {
                for instruction in [
                    Instruction::Load {
                        result: value.clone(),
                        buffer: buffer.clone(),
                        index: index.clone(),
                    },
                    Instruction::Store {
                        buffer: buffer.clone(),
                        index: index.clone(),
                        value: value.clone(),
                    },
                ] {
                    assert!(validate(&module(Place::Device, nested(vec![instruction]))).is_err());
                }
            }
        }
    }

    #[test]
    fn launches_check_place_target_and_argument_count() {
        let launch = Instruction::Launch {
            kernel: "kernel".into(),
            grid: "grid".into(),
            shared_bytes: "0".into(),
            arguments: vec![],
        };
        let mut host = module(Place::Host, nested(vec![launch.clone()]));
        assert!(validate(&host).is_err());
        host.kernels.insert(
            "kernel".into(),
            Function {
                symbol: "kernel".into(),
                inputs: vec![variable("arg", CType::U32)],
                outputs: vec![],
                body: vec![],
            },
        );
        assert!(validate(&host).is_err());
        host.kernels.get_mut("kernel").unwrap().inputs.clear();
        assert!(validate(&host).is_ok());
        host.kernels.get_mut("kernel").unwrap().body = nested(vec![launch]);
        assert!(validate(&host).is_err());
    }
}
