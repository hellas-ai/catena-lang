//! Execute compiled C backend functions by marshalling catena values into the
//! generated C ABI.

use thiserror::Error;

use super::compile::SharedObject;
use super::runtime::Value;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Unsupported value representation")]
    UnsupportedValue,
    #[error("Executor is not implemented yet")]
    Unimplemented,
}

/// Marshal runtime values into the generated C calling convention and invoke a
/// compiled function from the shared object.
pub(crate) fn exec<const M: usize, const N: usize>(
    artifact: &SharedObject,
    fn_name: &str,
    args: [Value; M],
) -> Result<[Value; N], ExecutorError> {
    let _ = (artifact, fn_name, args);
    Err(ExecutorError::Unimplemented)
}
