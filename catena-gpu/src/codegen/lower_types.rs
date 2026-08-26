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
        ("val", [inner]) => lower_runtime_type(inner),
        _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
    }
}
