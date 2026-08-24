use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Bool(u8),
    U32(u32),
    U64(u64),
    F32(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueKind {
    Bool,
    U32,
    U64,
    F32,
}

impl Value {
    pub(crate) fn kind(self) -> ValueKind {
        match self {
            Self::Bool(_) => ValueKind::Bool,
            Self::U32(_) => ValueKind::U32,
            Self::U64(_) => ValueKind::U64,
            Self::F32(_) => ValueKind::F32,
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value.into())
    }
}
impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}
impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}
impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}
