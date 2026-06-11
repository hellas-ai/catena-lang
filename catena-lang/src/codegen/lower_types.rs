use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

const VALUE_TYPES: &[&str] = &["val", "value"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweredType {
    Erased,
    Runtime(CType),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CType {
    Unit,
    Bool,
    U32,
    U64,
    F32,
    Pointer(Box<CType>),
    Named(String),
}

#[derive(Debug, Error)]
pub enum LowerTypeError {
    #[error("closure type `=>` survived representation lowering")]
    ClosureSurvived,
    #[error("runtime function pointer type `->` is not supported by this backend")]
    FunctionPointerRuntime,
    #[error("type `{0:?}` has no runtime representation")]
    NoRuntimeRepresentation(Tree<(), Operation>),
    #[error("type constructor `{name}` expected {expected} children, found {actual}")]
    InvalidArity {
        name: String,
        expected: usize,
        actual: usize,
    },
}

pub fn lower_type(ty: &Tree<(), Operation>) -> Result<LoweredType, LowerTypeError> {
    if contains_closure(ty) {
        return Err(LowerTypeError::ClosureSurvived);
    }

    if let Some(inner) = value_inner(ty)? {
        return lower_runtime_type(inner).map(LoweredType::Runtime);
    }

    match ty {
        Tree::Node(op, 0, children) if op.as_str() == ":" => {
            let [_name, ty] = expect_binary(op.as_str(), children)?;
            lower_runtime_type(ty).map(LoweredType::Runtime)
        }
        Tree::Node(op, 0, _children) if op.as_str() == "->" => {
            Err(LowerTypeError::FunctionPointerRuntime)
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.buf" => {
            let [element] = expect_unary(op.as_str(), children)?;
            Ok(LoweredType::Runtime(CType::Pointer(Box::new(
                lower_runtime_type(element)?,
            ))))
        }
        Tree::Node(op, 0, children) if op.as_str() == "buf" => {
            let [_cap, _len, element] = expect_ternary(op.as_str(), children)?;
            Ok(LoweredType::Runtime(CType::Pointer(Box::new(
                lower_runtime_type(element)?,
            ))))
        }
        Tree::Node(op, 0, children) if op.as_str() == "mem" => {
            let [_cap] = expect_unary(op.as_str(), children)?;
            Ok(LoweredType::Runtime(CType::Named(
                "catena_mem_t".to_string(),
            )))
        }
        Tree::Node(op, 0, children) if is_gpu_control_type(op.as_str()) && children.is_empty() => {
            Ok(LoweredType::Runtime(CType::Named(c_name_for_gpu_control(
                op.as_str(),
            ))))
        }
        _ => Ok(LoweredType::Erased),
    }
}

pub fn lower_runtime_type(ty: &Tree<(), Operation>) -> Result<CType, LowerTypeError> {
    if contains_closure(ty) {
        return Err(LowerTypeError::ClosureSurvived);
    }

    if let Some(inner) = value_inner(ty)? {
        return lower_runtime_type(inner);
    }

    match ty {
        Tree::Node(op, 0, children) if op.as_str() == ":" => {
            let [_name, ty] = expect_binary(op.as_str(), children)?;
            lower_runtime_type(ty)
        }
        Tree::Node(op, 0, children) if op.as_str() == "1" && children.is_empty() => Ok(CType::Unit),
        Tree::Node(op, 0, children) if op.as_str() == "bool" && children.is_empty() => {
            Ok(CType::Bool)
        }
        Tree::Node(op, 0, children) if op.as_str() == "u64" && children.is_empty() => {
            Ok(CType::U64)
        }
        Tree::Node(op, 0, children) if op.as_str() == "u32" && children.is_empty() => {
            Ok(CType::U32)
        }
        Tree::Node(op, 0, children) if op.as_str() == "f32" && children.is_empty() => {
            Ok(CType::F32)
        }
        Tree::Node(op, 0, _children) if op.as_str() == "->" => {
            Err(LowerTypeError::FunctionPointerRuntime)
        }
        Tree::Node(op, 0, children) if op.as_str() == "gpu.buf" => {
            let [element] = expect_unary(op.as_str(), children)?;
            Ok(CType::Pointer(Box::new(lower_runtime_type(element)?)))
        }
        Tree::Node(op, 0, children) if op.as_str() == "buf" => {
            let [_cap, _len, element] = expect_ternary(op.as_str(), children)?;
            Ok(CType::Pointer(Box::new(lower_runtime_type(element)?)))
        }
        Tree::Node(op, 0, children) if op.as_str() == "mem" => {
            let [_cap] = expect_unary(op.as_str(), children)?;
            Ok(CType::Named("catena_mem_t".to_string()))
        }
        Tree::Node(op, 0, children) if op.as_str() == "ix" => {
            let [_extent] = expect_unary(op.as_str(), children)?;
            Ok(CType::U64)
        }
        Tree::Node(op, 0, children) if is_gpu_control_type(op.as_str()) && children.is_empty() => {
            Ok(CType::Named(c_name_for_gpu_control(op.as_str())))
        }
        _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
    }
}

pub fn lower_interface(ty: &Tree<(), Operation>) -> Result<Vec<CType>, LowerTypeError> {
    let mut out = Vec::new();
    lower_interface_into(ty, &mut out)?;
    Ok(out)
}

fn lower_interface_into(
    ty: &Tree<(), Operation>,
    out: &mut Vec<CType>,
) -> Result<(), LowerTypeError> {
    match ty {
        Tree::Node(op, 0, children) if op.as_str() == "*" => {
            for child in children {
                lower_interface_into(child, out)?;
            }
        }
        other => {
            if let LoweredType::Runtime(ty) = lower_type(other)? {
                out.push(ty);
            }
        }
    }
    Ok(())
}

fn value_inner(ty: &Tree<(), Operation>) -> Result<Option<&Tree<(), Operation>>, LowerTypeError> {
    let Tree::Node(op, 0, children) = ty else {
        return Ok(None);
    };
    if !VALUE_TYPES.contains(&op.as_str()) {
        return Ok(None);
    }
    let [inner] = expect_unary(op.as_str(), children)?;
    Ok(Some(inner))
}

fn contains_closure(ty: &Tree<(), Operation>) -> bool {
    match ty {
        Tree::Node(op, _, children) => op.as_str() == "=>" || children.iter().any(contains_closure),
        _ => false,
    }
}

fn expect_unary<'a>(
    name: &str,
    children: &'a [Tree<(), Operation>],
) -> Result<[&'a Tree<(), Operation>; 1], LowerTypeError> {
    match children {
        [only] => Ok([only]),
        _ => Err(LowerTypeError::InvalidArity {
            name: name.to_string(),
            expected: 1,
            actual: children.len(),
        }),
    }
}

fn expect_binary<'a>(
    name: &str,
    children: &'a [Tree<(), Operation>],
) -> Result<[&'a Tree<(), Operation>; 2], LowerTypeError> {
    match children {
        [first, second] => Ok([first, second]),
        _ => Err(LowerTypeError::InvalidArity {
            name: name.to_string(),
            expected: 2,
            actual: children.len(),
        }),
    }
}

fn expect_ternary<'a>(
    name: &str,
    children: &'a [Tree<(), Operation>],
) -> Result<[&'a Tree<(), Operation>; 3], LowerTypeError> {
    match children {
        [first, second, third] => Ok([first, second, third]),
        _ => Err(LowerTypeError::InvalidArity {
            name: name.to_string(),
            expected: 3,
            actual: children.len(),
        }),
    }
}

fn is_gpu_control_type(name: &str) -> bool {
    matches!(
        name,
        "gpu.3d" | "gpu.env" | "gpu.launch_params" | "gpu.state"
    )
}

fn c_name_for_gpu_control(name: &str) -> String {
    match name {
        "gpu.3d" => "catena_dim3_t",
        "gpu.env" => "catena_gpu_env_t",
        "gpu.launch_params" => "catena_launch_params_t",
        "gpu.state" => "catena_gpu_state_t",
        _ => unreachable!("checked by is_gpu_control_type"),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str, children: Vec<Tree<(), Operation>>) -> Tree<(), Operation> {
        Tree::Node(name.parse().unwrap(), 0, children)
    }

    fn leaf(index: usize) -> Tree<(), Operation> {
        Tree::Leaf(index, ())
    }

    #[test]
    fn plain_type_level_objects_erase() {
        assert_eq!(
            lower_type(&node("f32", vec![])).unwrap(),
            LoweredType::Erased
        );
        assert_eq!(lower_type(&leaf(0)).unwrap(), LoweredType::Erased);
    }

    #[test]
    fn value_wrapped_scalars_lower_to_runtime_types() {
        assert_eq!(
            lower_type(&node("val", vec![node("bool", vec![])])).unwrap(),
            LoweredType::Runtime(CType::Bool)
        );
        assert_eq!(
            lower_type(&node("val", vec![node("f32", vec![])])).unwrap(),
            LoweredType::Runtime(CType::F32)
        );
    }

    #[test]
    fn products_flatten_and_erase_type_level_components() {
        let ty = node(
            "*",
            vec![
                node("gpu.env", vec![]),
                leaf(0),
                node("val", vec![node("bool", vec![])]),
            ],
        );
        assert_eq!(
            lower_interface(&ty).unwrap(),
            vec![CType::Named("catena_gpu_env_t".to_string()), CType::Bool]
        );
    }

    #[test]
    fn runtime_function_pointers_are_rejected() {
        let ty = node(
            "val",
            vec![node(
                "->",
                vec![
                    node("*", vec![leaf(0), node("val", vec![node("bool", vec![])])]),
                    node("val", vec![node("u64", vec![])]),
                ],
            )],
        );
        assert!(matches!(
            lower_type(&ty),
            Err(LowerTypeError::FunctionPointerRuntime)
        ));
    }

    #[test]
    fn closures_are_errors() {
        let ty = node("=>", vec![node("bool", vec![]), node("bool", vec![])]);
        assert!(matches!(
            lower_type(&ty),
            Err(LowerTypeError::ClosureSurvived)
        ));
    }
}
