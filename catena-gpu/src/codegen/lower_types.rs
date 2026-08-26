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
    U64Ptr,
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
        "gpu.scheduling" => lower_runtime_type(ty).map(LoweredType::Runtime),
        _ => Ok(LoweredType::Erased),
    }
}

fn lower_runtime_type(ty: &Tree<(), Operation>) -> Result<CType, LowerTypeError> {
    let Tree::Node(operation, _, children) = ty else {
        return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
    };
    match (operation.as_str(), children.as_slice()) {
        ("bool", []) => Ok(CType::Bool),
        ("u32", []) => Ok(CType::U32),
        ("u64", []) => Ok(CType::U64),
        ("f32", []) => Ok(CType::F32),
        ("gpu.grid", [_grid_shape, _block_shape, _global_shape, _global_size]) => Ok(CType::Grid),
        ("mem", [_capability]) => Ok(CType::MemOwn),
        ("gpu.global", [_capability, _buffer_id, _buffer_size, element])
            if is_nullary(element, "u64") =>
        {
            Ok(CType::U64Ptr)
        }
        ("gpu.thread", [_grid]) => Ok(CType::Thread),
        ("gpu.block", [_grid, _block_id]) => Ok(CType::Block),
        ("gpu.scheduling", [_buffer_id, _buffer_size, _grid, _schedule]) => Ok(CType::Scheduling),
        ("ix", [_shape]) => Ok(CType::U64),
        ("val", [inner]) => lower_runtime_type(inner),
        _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
    }
}

fn is_nullary(tree: &Tree<(), Operation>, name: &str) -> bool {
    matches!(tree, Tree::Node(operation, _, children) if operation.as_str() == name && children.is_empty())
}
