use super::{
    executor::{AbiValue, ArgValue},
    mem::Mem,
};

/// Public Catena runtime values accepted at program boundaries.
#[derive(Debug)]
pub enum Value {
    Bool(u8),
    U64(u64),
    Mem(Mem),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Bool,
    U64,
    Mem,
}

impl Value {
    pub fn bool(value: bool) -> Self {
        Value::Bool(u8::from(value))
    }

    pub fn u64(value: u64) -> Self {
        Value::U64(value)
    }

    pub(crate) fn kind(&self) -> ValueKind {
        match self {
            Value::Bool(_) => ValueKind::Bool,
            Value::U64(_) => ValueKind::U64,
            Value::Mem(_) => ValueKind::Mem,
        }
    }
}

impl Value {
    pub(crate) fn as_input_arg(&self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Val(AbiValue::U8(value)),
            Value::U64(value) => ArgValue::Val(AbiValue::U64(value)),
            Value::Mem(value) => ArgValue::Val(AbiValue::Mem(&value.abi)),
        }
    }

    pub(crate) fn as_output_arg(&mut self) -> ArgValue<'_> {
        match self {
            Value::Bool(value) => ArgValue::Out(AbiValue::U8(value)),
            Value::U64(value) => ArgValue::Out(AbiValue::U64(value)),
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
