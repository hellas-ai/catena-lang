use super::executor::ArgValue;

/// Public catena runtime values for the C backend.
#[derive(Debug)]
pub enum Value {
    Extent(u64),
    Index(u64),
    F32(f32),
}

/// Semantic kinds of public runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    Extent,
    Index,
    F32,
}

impl Value {
    pub(crate) fn kind(&self) -> ValueKind {
        match self {
            Value::Extent(_) => ValueKind::Extent,
            Value::Index(_) => ValueKind::Index,
            Value::F32(_) => ValueKind::F32,
        }
    }
}

impl Value {
    pub(crate) fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Extent => Value::Extent(0),
            ValueKind::Index => Value::Index(0),
            ValueKind::F32 => Value::F32(0.0),
        }
    }

    pub(crate) fn as_input_arg(&self) -> ArgValue<'_> {
        match self {
            Value::Extent(value) | Value::Index(value) => ArgValue::U64(value),
            Value::F32(value) => ArgValue::F32(value),
        }
    }

    pub(crate) fn as_output_arg(&mut self) -> ArgValue<'_> {
        match self {
            Value::Extent(value) | Value::Index(value) => ArgValue::OutU64(value),
            Value::F32(value) => ArgValue::OutF32(value),
        }
    }
}
