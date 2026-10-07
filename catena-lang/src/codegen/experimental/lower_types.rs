//! Representation rules for experimental types. Unknown runtime types are errors.
use super::{CodegenError, values::Obj};
use crate::runtime::ValueKind;
use metacat::tree::Tree;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CType {
    Bool,
    U32,
    U64,
    F32,
    MemOwn,
    MemRef,
    Grid,
    Block,
    Thread,
    Index,
    Global(Box<CType>),
    Shared(Box<CType>),
    Layout(usize),
}

pub(super) fn node(name: &str, children: Vec<Obj>) -> Obj {
    Tree::Node(name.parse().expect("internal type name"), 0, children)
}

pub(super) fn children<'a>(ty: &'a Obj, name: &str) -> Option<&'a [Obj]> {
    match ty {
        Tree::Node(op, 0, args) if op.as_str() == name => Some(args),
        _ => None,
    }
}

pub(super) fn runtime(ty: &Obj) -> Result<Option<CType>, CodegenError> {
    if let Some([inner]) = children(ty, "val") {
        return concrete(inner).map(Some);
    }
    if let Some([_, inner]) = children(ty, ":") {
        return concrete(inner).map(Some);
    }
    if children(ty, "*").is_some() {
        return Err(CodegenError::Type(
            "product must be lowered field by field".into(),
        ));
    }
    Ok(None)
}

pub(super) fn concrete(ty: &Obj) -> Result<CType, CodegenError> {
    let Tree::Node(op, 0, args) = ty else {
        return Err(CodegenError::NoRuntimeRepresentation(ty.clone()));
    };
    let result = match (op.as_str(), args.as_slice()) {
        ("bool", []) => CType::Bool,
        ("u32", []) => CType::U32,
        ("u64", []) => CType::U64,
        ("f32", []) => CType::F32,
        ("mem", [cap]) if children(cap, "cap.own") == Some(&[]) => CType::MemOwn,
        ("mem", [cap]) if children(cap, "cap.ref") == Some(&[]) => CType::MemRef,
        ("stdlib.gpu.geometry.type.Grid", [_, _]) => CType::Grid,
        ("stdlib.gpu.geometry.type.Block", [_, _, _]) => CType::Block,
        ("stdlib.gpu.geometry.type.Thread", [_, _, _, _]) => CType::Thread,
        ("stdlib.gpu.geometry.type.Index", [_]) => CType::Index,
        ("stdlib.gpu.memory.type.Ix" | "stdlib.gpu.memory.type.SlotName", [_]) => CType::U64,
        ("stdlib.gpu.memory.type.Global", [_, element]) => {
            CType::Global(Box::new(concrete(element)?))
        }
        ("stdlib.gpu.memory.type.Shared", [_, _, _, _, _, _, _, _, element]) => {
            CType::Shared(Box::new(concrete(element)?))
        }
        ("stdlib.gpu.memory.type.SharedLayout", [layout])
        | ("stdlib.gpu.memory.type.SharedMemory", [_, _, _, layout]) => {
            CType::Layout(layout_len(layout)?)
        }
        _ => {
            return Err(CodegenError::Type(format!(
                "unsupported experimental runtime type {ty:?}"
            )));
        }
    };
    Ok(result)
}

fn layout_len(ty: &Obj) -> Result<usize, CodegenError> {
    if children(ty, "list.Nil") == Some(&[]) {
        return Ok(0);
    }
    if let Some([_, tail]) = children(ty, "list.Cons") {
        return Ok(1 + layout_len(tail)?);
    }
    Err(CodegenError::Type(format!(
        "shared layout must have a statically known slot list: {ty:?}"
    )))
}

impl CType {
    pub(super) fn c_name(&self) -> String {
        match self {
            Self::Bool => "uint8_t".into(),
            Self::U32 => "uint32_t".into(),
            Self::U64 => "uint64_t".into(),
            Self::F32 => "float".into(),
            Self::MemOwn => "catena_mem_own_t".into(),
            Self::MemRef => "catena_mem_ref_t".into(),
            Self::Grid => "catena_grid".into(),
            Self::Block => "catena_block".into(),
            Self::Thread => "catena_thread".into(),
            Self::Index => "catena_index".into(),
            Self::Global(t) => format!("catena_global<{}>", t.c_name()),
            Self::Shared(t) => format!("catena_shared_view<{}>", t.c_name()),
            Self::Layout(n) => format!("catena_layout<{n}>"),
        }
    }
    pub(super) fn abi(&self) -> Option<ValueKind> {
        Some(match self {
            Self::Bool => ValueKind::Bool,
            Self::U32 => ValueKind::U32,
            Self::U64 => ValueKind::U64,
            Self::F32 => ValueKind::F32,
            Self::MemOwn => ValueKind::MemOwn,
            Self::MemRef => ValueKind::MemRef,
            _ => return None,
        })
    }
    pub(super) fn numeric(&self) -> bool {
        matches!(self, Self::U32 | Self::U64 | Self::F32)
    }
    pub(super) fn integral(&self) -> bool {
        matches!(self, Self::U32 | Self::U64)
    }
}
