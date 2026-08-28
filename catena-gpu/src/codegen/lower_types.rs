use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweredType {
    Erased,
    Runtime(CType),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
    #[error("runtime type variable {0} was not specialized")]
    Unspecialized(usize),
    #[error("runtime type variable {variable} was inferred as both `{first:?}` and `{second:?}`")]
    ConflictingSpecialization {
        variable: usize,
        first: CType,
        second: CType,
    },
    #[error("runtime type `{declared:?}` does not match concrete type `{target:?}`")]
    SpecializationMismatch {
        declared: Tree<(), Operation>,
        target: CType,
    },
}

pub fn lower_type(
    ty: &Tree<(), Operation>,
    substitutions: &BTreeMap<usize, CType>,
) -> Result<LoweredType, LowerTypeError> {
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
            lower_runtime_type(inner, substitutions).map(LoweredType::Runtime)
        }
        ":" => {
            let [_name, inner] = children.as_slice() else {
                return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
            };
            lower_runtime_type(inner, substitutions).map(LoweredType::Runtime)
        }
        "mem" => lower_runtime_type(ty, substitutions).map(LoweredType::Runtime),
        _ => Ok(LoweredType::Erased),
    }
}

fn lower_runtime_type(
    ty: &Tree<(), Operation>,
    substitutions: &BTreeMap<usize, CType>,
) -> Result<CType, LowerTypeError> {
    let Tree::Node(operation, _, children) = ty else {
        let Tree::Leaf(index, ()) = ty else {
            return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
        };
        return substitutions
            .get(index)
            .cloned()
            .ok_or(LowerTypeError::Unspecialized(*index));
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
            let element = Box::new(lower_runtime_type(element, substitutions)?);
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
        ("val", [inner]) => lower_runtime_type(inner, substitutions),
        _ => Err(LowerTypeError::NoRuntimeRepresentation(ty.clone())),
    }
}

pub fn infer_type(
    ty: &Tree<(), Operation>,
    concrete: &CType,
    substitutions: &mut BTreeMap<usize, CType>,
) -> Result<(), LowerTypeError> {
    let runtime = match ty {
        Tree::Node(operation, _, children) if operation.as_str() == "val" => {
            let [inner] = children.as_slice() else {
                return Err(LowerTypeError::InvalidArity {
                    name: "val".into(),
                    actual: children.len(),
                });
            };
            inner
        }
        Tree::Node(operation, _, children) if operation.as_str() == ":" => {
            let [_name, inner] = children.as_slice() else {
                return Err(LowerTypeError::NoRuntimeRepresentation(ty.clone()));
            };
            inner
        }
        Tree::Node(operation, _, _) if operation.as_str() == "mem" => ty,
        _ => return Ok(()),
    };
    infer_runtime_type(runtime, concrete, substitutions)
}

fn infer_runtime_type(
    ty: &Tree<(), Operation>,
    concrete: &CType,
    substitutions: &mut BTreeMap<usize, CType>,
) -> Result<(), LowerTypeError> {
    if let Tree::Leaf(variable, ()) = ty {
        if let Some(first) = substitutions.get(variable) {
            if first != concrete {
                return Err(LowerTypeError::ConflictingSpecialization {
                    variable: *variable,
                    first: first.clone(),
                    second: concrete.clone(),
                });
            }
        } else {
            substitutions.insert(*variable, concrete.clone());
        }
        return Ok(());
    }

    let Tree::Node(operation, _, children) = ty else {
        return Err(LowerTypeError::SpecializationMismatch {
            declared: ty.clone(),
            target: concrete.clone(),
        });
    };
    match (operation.as_str(), children.as_slice(), concrete) {
        ("val", [inner], _) => infer_runtime_type(inner, concrete, substitutions),
        ("gpu.global", [capability, _size, element], CType::Ptr(actual))
            if is_capability(capability, "cap.own") =>
        {
            infer_runtime_type(element, actual, substitutions)
        }
        ("gpu.global", [capability, _size, element], CType::ConstPtr(actual))
            if is_capability(capability, "cap.ref") =>
        {
            infer_runtime_type(element, actual, substitutions)
        }
        _ => match lower_runtime_type(ty, substitutions) {
            Ok(actual) if &actual == concrete => Ok(()),
            _ => Err(LowerTypeError::SpecializationMismatch {
                declared: ty.clone(),
                target: concrete.clone(),
            }),
        },
    }
}

fn is_capability(ty: &Tree<(), Operation>, expected: &str) -> bool {
    matches!(ty, Tree::Node(operation, _, children) if operation.as_str() == expected && children.is_empty())
}
