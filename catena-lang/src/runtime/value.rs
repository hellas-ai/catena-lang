use super::mem::{MemOwn, MemRef};
use serde::{Deserialize, Serialize};

/// Public Catena runtime values accepted at program boundaries.
#[derive(Debug)]
pub enum Value<'a> {
    Bool(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    MemOwn(MemOwn),
    MemRef(MemRef<'a>),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueKind {
    Bool,
    U16,
    U32,
    U64,
    F32,
    MemOwn,
    MemRef,
}

impl<'a> Value<'a> {
    pub fn bool(value: bool) -> Self {
        Value::Bool(u8::from(value))
    }

    pub fn u64(value: u64) -> Self {
        Value::U64(value)
    }

    pub fn u16(value: u16) -> Self {
        Value::U16(value)
    }

    pub fn u32(value: u32) -> Self {
        Value::U32(value)
    }

    pub fn f32(value: f32) -> Self {
        Value::F32(value)
    }

    pub(super) fn kind(&self) -> ValueKind {
        match self {
            Value::Bool(_) => ValueKind::Bool,
            Value::U16(_) => ValueKind::U16,
            Value::U32(_) => ValueKind::U32,
            Value::U64(_) => ValueKind::U64,
            Value::F32(_) => ValueKind::F32,
            Value::MemOwn(_) => ValueKind::MemOwn,
            Value::MemRef(_) => ValueKind::MemRef,
        }
    }
}

impl<'a> From<bool> for Value<'a> {
    fn from(value: bool) -> Self {
        Value::bool(value)
    }
}

impl<'a> From<u64> for Value<'a> {
    fn from(value: u64) -> Self {
        Value::u64(value)
    }
}

impl<'a> From<u16> for Value<'a> {
    fn from(value: u16) -> Self {
        Value::u16(value)
    }
}

impl<'a> From<u32> for Value<'a> {
    fn from(value: u32) -> Self {
        Value::u32(value)
    }
}

impl<'a> From<f32> for Value<'a> {
    fn from(value: f32) -> Self {
        Value::f32(value)
    }
}

impl From<MemOwn> for Value<'static> {
    fn from(value: MemOwn) -> Self {
        Value::MemOwn(value)
    }
}

impl<'a> From<MemRef<'a>> for Value<'a> {
    fn from(value: MemRef<'a>) -> Self {
        Value::MemRef(value)
    }
}
