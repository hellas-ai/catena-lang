use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweredType {
    Erased,
    Runtime(CType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    Bool,
    U32,
    U64,
    F32,
    Grid,
    MemOwn,
    MemRef,
    Ptr(Box<CType>),
    ConstPtr(Box<CType>),
    Generic(usize),
    Ix,
    Thread,
    Block,
    Scheduling,
}

#[derive(Debug, Error)]
pub enum LowerTypeError {
    #[error("type `{0:?}` has no runtime representation")]
    NoRuntimeRepresentation(Tree<(), Operation>),
    #[error("type constructor `{name}` expected one child, found {actual}")]
    InvalidArity { name: String, actual: usize },
}

pub fn lower_type(ty: &Tree<(), Operation>) -> Result<LoweredType, LowerTypeError> {
    let Tree::Node(operation, _, children) = ty else {
        return Ok(LoweredType::Erased);
    };
    match operation.as_str() {
        "val" => {
            let [inner] = children.as_slice() else {
                return Err(LowerTypeError::InvalidArity {
                    name: "val".into(),
                    actual: children.len(),
                });
            };
            lower_runtime_type(inner).map(LoweredType::Runtime)
        }
        ":" => {
            let [_name, inner] = children.as_slice() else {
                return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
            };
            lower_runtime_type(inner).map(LoweredType::Runtime)
        }
        "mem" => lower_runtime_type(ty).map(LoweredType::Runtime),
        _ => Ok(LoweredType::Erased),
    }
}

fn lower_runtime_type(ty: &Tree<(), Operation>) -> Result<CType, LowerTypeError> {
    let Tree::Node(operation, _, children) = ty else {
        let Tree::Leaf(index, ()) = ty else {
            return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
        };
        return Ok(CType::Generic(*index));
    };
    match (operation.as_str(), children.as_slice()) {
        ("bool", []) => Ok(CType::Bool),
        ("u32", []) => Ok(CType::U32),
        ("positive-u32", []) => Ok(CType::U32),
        ("u64", []) => Ok(CType::U64),
        ("f32", []) => Ok(CType::F32),
        ("gpu.grid", [_grid_shape, _block_shape, _global_shape]) => Ok(CType::Grid),
        ("mem", [capability]) => match capability {
            Tree::Node(operation, _, children)
                if operation.as_str() == "cap.own" && children.is_empty() =>
            {
                Ok(CType::MemOwn)
            }
            Tree::Node(operation, _, children)
                if operation.as_str() == "cap.ref" && children.is_empty() =>
            {
                Ok(CType::MemRef)
            }
            _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
        },
        ("gpu.global", [capability, _buffer_size, element]) => {
            let element = Box::new(lower_runtime_type(element)?);
            match capability {
                Tree::Node(operation, _, children)
                    if operation.as_str() == "cap.own" && children.is_empty() =>
                {
                    Ok(CType::Ptr(element))
                }
                Tree::Node(operation, _, children)
                    if operation.as_str() == "cap.ref" && children.is_empty() =>
                {
                    Ok(CType::ConstPtr(element))
                }
                _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
            }
        }
        ("gpu.thread", [_grid]) => Ok(CType::Thread),
        ("gpu.block", [_grid, _block_name]) => Ok(CType::Block),
        ("gpu.scheduling", [_buffer_name, _buffer_size, _grid, _schedule_name]) => {
            Ok(CType::Scheduling)
        }
        ("ix", [_shape]) => Ok(CType::Ix),
        ("val", [inner]) => lower_runtime_type(inner),
        _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
    }
}
