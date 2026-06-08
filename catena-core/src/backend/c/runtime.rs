use thiserror::Error;

use super::compile::{CompileError, SharedObject};
use super::executor::{CallFrame, ExecutorError};
use super::value::{Value, ValueKind};
use metacat::theory::{Theory, TheorySet};

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: SharedObject,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("Failed to parse program: {0}")]
    Parse(#[from] metacat::theory::load::LoadError),
    #[error("No runtime theory found")]
    NoRuntimeTheory,
    #[error(transparent)]
    Compile(#[from] CompileError),
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Argument {index} expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        index: usize,
        expected: ValueKind,
        actual: ValueKind,
    },
    #[error("Function '{name}' expected {expected} inputs, got {actual}")]
    InputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Function '{name}' expected {expected} outputs, got {actual}")]
    OutputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Executor error: {0}")]
    Executor(#[from] ExecutorError),
}

impl Runtime {
    pub fn new(source: &str) -> Result<Runtime, InitError> {
        let theory_set = TheorySet::from_text(source)?;
        let theory = runtime_theory(&theory_set).ok_or(InitError::NoRuntimeTheory)?;
        let artifact = super::compile::compile(theory)?;
        Ok(Self { artifact })
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let signature = self
            .artifact
            .signature(fn_name)
            .ok_or_else(|| ExecError::UnknownFunction(fn_name.to_string()))?;

        if signature.inputs.len() != M {
            return Err(ExecError::InputArityMismatch {
                name: fn_name.to_string(),
                expected: signature.inputs.len(),
                actual: M,
            });
        }
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: fn_name.to_string(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }

        let input_values = collect_inputs(&args, signature)?;
        let mut output_values: Vec<Value> = signature
            .outputs
            .iter()
            .copied()
            .map(Value::zeroed)
            .collect();

        let mut frame_args = Vec::with_capacity(M + N);
        for value in &input_values {
            frame_args.push(value.as_input_arg());
        }
        for value in &mut output_values {
            frame_args.push(value.as_output_arg());
        }

        super::executor::exec(
            self.artifact.path(),
            &signature.symbol,
            CallFrame {
                args: &mut frame_args,
            },
        )?;

        Ok(output_values
            .try_into()
            .expect("output arity already validated"))
    }
}

fn runtime_theory(theory_set: &TheorySet) -> Option<&Theory> {
    // Legacy runtime selection shim: compile the first user theory in the file.
    // This should be removed soon in favor of an explicit runtime theory choice.
    theory_set
        .theories
        .values()
        .find(|theory| matches!(theory, Theory::Theory { .. }))
}

fn collect_inputs<const M: usize>(
    args: &[Value; M],
    signature: &super::compile::FunctionSignature,
) -> Result<Vec<Value>, ExecError> {
    args.iter()
        .zip(signature.inputs.iter().copied())
        .enumerate()
        .map(|(index, (value, expected))| validate_input(index, value, expected))
        .collect()
}

// Verify that an input value has the expected kind, but does *not* check deeply (e.g., that an
// index belongs to its declared extent.
fn validate_input(index: usize, value: &Value, expected: ValueKind) -> Result<Value, ExecError> {
    match (value, expected) {
        (Value::Extent(value), ValueKind::Extent) => Ok(Value::Extent(*value)),
        (Value::Index(value), ValueKind::Index) => Ok(Value::Index(*value)),
        (Value::F32(value), ValueKind::F32) => Ok(Value::F32(*value)),
        _ => Err(ExecError::TypeMismatch {
            index,
            expected,
            actual: value.kind(),
        }),
    }
}
