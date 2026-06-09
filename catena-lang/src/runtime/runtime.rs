use thiserror::Error;

use std::{path::PathBuf, sync::Arc};

use libloading::Library;

use super::artifact::{Artifact, ArtifactError};
use super::executor::{CallFrame, ExecutorError};
use super::mem::{Hip, Mem, MemError};
use super::{
    signature::{SignatureTable, signatures},
    value::{Value, ValueKind},
};
use crate::compile::CompileFailure;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    // Keep the tempdir-backed shared object alive for as long as the library is loaded.
    _artifact: Artifact,
    /// The loaded shared object
    library: Library,
    /// A handle to the loaded hip .so file, which we call for allocating memory
    hip: Arc<Hip>,
    /// Function signatures (runtime Rust ↔ C typechecking)
    signatures: SignatureTable,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("Failed to parse program: {0}")]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error(transparent)]
    Compile(#[from] CompileFailure),
    #[error("compile report did not contain GPU modules")]
    MissingGpuModules,
    #[error("failed to write generated report files: {0}")]
    DumpReport(#[from] std::io::Error),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("failed to load compiled shared object: {0}")]
    LoadLibrary(#[source] libloading::Error),
    #[error(transparent)]
    Mem(#[from] MemError),
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Unknown source function '{0}'")]
    UnknownSourceFunction(String),
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
    /// Construct a new runtime from a list of paths, interpreted as catena programs (&stdlib)
    pub fn new<I>(paths: I) -> Result<Runtime, InitError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let raw_theories = metacat::theory::RawTheorySet::from_files(paths)?;
        let report = crate::compile::compile(raw_theories)?;
        let modules = report
            .gpu_modules
            .as_ref()
            .ok_or(InitError::MissingGpuModules)?;
        let signature_table = signatures(modules);

        let report_dir = tempfile::Builder::new()
            .prefix("catena-report-")
            .tempdir()?;
        report.dump_to_dir(report_dir.path())?;
        let cpp_path = report_dir.path().join("gpu/program.cpp");
        let artifact = super::artifact::compile(&cpp_path)?;
        let library = unsafe { Library::new(artifact.path()) }.map_err(InitError::LoadLibrary)?;
        let hip = Arc::new(Hip::load()?);

        Ok(Self {
            _artifact: artifact,
            library,
            hip,
            signatures: signature_table,
        })
    }

    /// Look up the generated C symbol for a source-level `program` definition name.
    pub fn symbol(&self, name: &str) -> Option<&str> {
        self.signatures.source_symbols.get(name).map(String::as_str)
    }

    pub fn mem_u64(&self, values: &[u64]) -> Result<Value, MemError> {
        Mem::from_u64_slice(self.hip.clone(), values).map(Value::Mem)
    }

    /// Run a source-level `program` definition, which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let symbol = self
            .symbol(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        self.exec_symbol(symbol, args)
    }

    /// Run the generated C symbol, which must have M arguments, and return its N arguments.
    pub fn exec_symbol<const M: usize, const N: usize>(
        &self,
        symbol: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let signature = self
            .signatures
            .functions
            .get(symbol)
            .ok_or_else(|| ExecError::UnknownFunction(symbol.to_string()))?;

        // Check arity/coarity lines up with what's in the function signature.
        if signature.inputs.len() != M {
            return Err(ExecError::InputArityMismatch {
                name: symbol.to_string(),
                expected: signature.inputs.len(),
                actual: M,
            });
        }
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: symbol.to_string(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }

        let mut output_values: Vec<Value> = signature
            .outputs
            .iter()
            .copied()
            .map(|kind| self.zeroed_value(kind))
            .collect();

        // Construct the `ArgValue`s with which to call the function
        let mut frame_args = Vec::with_capacity(M + N);
        for (index, (value, expected)) in args
            .iter()
            .zip(signature.inputs.iter().copied())
            .enumerate()
        {
            if value.kind() != expected {
                return Err(ExecError::TypeMismatch {
                    index,
                    expected,
                    actual: value.kind(),
                });
            }
            frame_args.push(value.as_input_arg());
        }
        for value in &mut output_values {
            frame_args.push(value.as_output_arg());
        }

        super::executor::exec(
            &self.library,
            symbol,
            CallFrame {
                args: &mut frame_args,
            },
        )?;

        Ok(output_values
            .try_into()
            .expect("output arity already validated"))
    }

    fn zeroed_value(&self, kind: ValueKind) -> Value {
        match kind {
            ValueKind::Bool => Value::Bool(0),
            ValueKind::U64 => Value::U64(0),
            ValueKind::Mem => Value::Mem(Mem::null(self.hip.clone())),
        }
    }
}
