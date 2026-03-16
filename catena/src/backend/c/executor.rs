//! Execute compiled C backend functions through a small ABI-oriented interface.

use thiserror::Error;

use super::compile::SharedObject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgType {
    U64,
    F32,
}

#[derive(Debug)]
pub enum ArgValue<'a> {
    U64(&'a u64),
    F32(&'a f32),
    OutU64(&'a mut u64),
    OutF32(&'a mut f32),
}

impl ArgValue<'_> {
    pub fn arg_type(&self) -> ArgType {
        match self {
            ArgValue::U64(_) | ArgValue::OutU64(_) => ArgType::U64,
            ArgValue::F32(_) | ArgValue::OutF32(_) => ArgType::F32,
        }
    }

    pub fn is_output(&self) -> bool {
        matches!(self, ArgValue::OutU64(_) | ArgValue::OutF32(_))
    }
}

#[derive(Debug)]
pub struct CallFrame<'a> {
    pub args: &'a mut [ArgValue<'a>],
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Executor is not implemented yet")]
    Unimplemented,
}

/// Invoke a compiled symbol using the generated C ABI.
///
/// The executor only knows about ABI-level scalar slots and output pointers.
/// Catena-specific type mapping belongs in `runtime`.
pub(crate) fn exec(
    artifact: &SharedObject,
    fn_name: &str,
    frame: CallFrame<'_>,
) -> Result<(), ExecutorError> {
    let _ = (artifact, frame);
    Err(ExecutorError::UnknownFunction(fn_name.to_string()))
}
