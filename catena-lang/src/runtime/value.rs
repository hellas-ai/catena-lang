use super::{
    executor::{AbiValue, ArgValue},
    mem::Mem,
};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Value {
    pub(crate) fn as_input_arg(&self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Val(AbiValue::U8(value)),
            Value::U32(value) => ArgValue::Val(AbiValue::U32(value)),
            Value::U64(value) => ArgValue::Val(AbiValue::U64(value)),
            Value::F32(value) => ArgValue::Val(AbiValue::F32(value)),
            Value::Mem(value) => ArgValue::Val(AbiValue::Mem(&value.abi)),
        }
    }

    pub(crate) fn as_output_arg(&mut self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Out(AbiValue::U8(value)),
            Value::U32(value) => ArgValue::Out(AbiValue::U32(value)),
            Value::U64(value) => ArgValue::Out(AbiValue::U64(value)),
            Value::F32(value) => ArgValue::Out(AbiValue::F32(value)),
            Value::Mem(value) => ArgValue::Out(AbiValue::Mem(&mut value.abi)),
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
