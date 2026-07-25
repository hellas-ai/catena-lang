use super::mem::Mem;
use serde::{Deserialize, Serialize};

/// Public Catena runtime values accepted at program boundaries.
#[derive(Debug)]
pub enum Value {
    Bool(u8),
    U32(u32),
    U64(u64),
    F32(f32),
    Mem(Mem),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueKind {
    Bool,
    U32,
    U64,
    F32,
    Mem,
}

impl Value {
    pub fn bool(value: bool) -> Self {
        Value::Bool(u8::from(value))
    }

    pub fn u64(value: u64) -> Self {
        Value::U64(value)
    }

    pub fn u32(value: u32) -> Self {
        Value::U32(value)
    }

    pub fn f32(value: f32) -> Self {
        Value::F32(value)
    }

    pub(crate) fn kind(&self) -> ValueKind {
        match self {
            Value::Bool(_) => ValueKind::Bool,
            Value::U32(_) => ValueKind::U32,
            Value::U64(_) => ValueKind::U64,
            Value::F32(_) => ValueKind::F32,
            Value::Mem(_) => ValueKind::Mem,
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::bool(value)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Value::u64(value)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Value::u32(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::f32(value)
    }
}
