use thiserror::Error;

use super::compile::{CompileError, SharedObject};
use super::executor::ExecutorError;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: SharedObject,
}

/// Public interface for marshalling values into/out of the runtime
#[derive(Debug)]
pub enum Value {
    Extent(usize),
}

// An opaque pointer to a value reference.
// NOTE: these *cannot* be safely copied; we rely on them being 'consumable'.
#[derive(Debug)]
pub struct ValueRef {
    value: Value,
}

pub type InitError = CompileError;

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("Executor error: {0}")]
    Executor(#[from] ExecutorError),
}

impl Runtime {
    pub fn new(source: &str) -> Result<Runtime, InitError> {
        let artifact = super::compile::compile(source)?;
        Ok(Self { artifact })
    }

    /// Move a value into the runtime
    pub fn value(&self, value: Value) -> ValueRef {
        ValueRef { value }
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [ValueRef; M],
    ) -> Result<[ValueRef; N], ExecError> {
        let args = args.map(|arg| arg.value);
        let outputs = super::executor::exec(&self.artifact, fn_name, args)?;
        Ok(outputs.map(|value| ValueRef { value }))
    }
}
