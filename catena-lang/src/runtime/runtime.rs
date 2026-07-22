use thiserror::Error;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use libloading::Library;
use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};

use super::artifact::{Artifact, ArtifactError};
use super::executor::{Executor, ExecutorError};
use super::mem::{GpuRuntime, Mem, MemError};
use super::{
    signature::{FunctionSignature, SignatureTable, signatures},
    value::{Value, ValueKind},
};
use crate::codegen::{GpuDialect, gpu::GpuRenderError, gpu::render_modules};
use crate::compile::CompileFailure;
use metacat::theory::RawTheorySet;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    // Keep the tempdir-backed shared object alive for as long as the library is loaded.
    _artifact: Artifact,
    /// Prepared entry points in the loaded shared object.
    executor: Executor,
    /// A handle to the loaded GPU runtime library, which we call for allocating memory.
    gpu: Arc<GpuRuntime>,
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
    #[error("failed to render generated {dialect:?} source: {source}")]
    RenderGpu {
        dialect: GpuDialect,
        #[source]
        source: GpuRenderError,
    },
    #[error("failed to write generated GPU source to {path}: {source}")]
    WriteGeneratedSource {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create generated GPU build directory {path}: {source}")]
    CreateBuildDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("failed to load compiled shared object {path}: {source}")]
    LoadLibrary {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
    #[error(transparent)]
    Mem(#[from] MemError),
}

#[derive(Debug, Error)]
pub enum ExecError {
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
}

impl Runtime {
    /// Construct a new runtime from a list of paths, interpreted as catena programs (&stdlib)
    pub fn new<I>(paths: I, dialect: GpuDialect) -> Result<Runtime, InitError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let raw_theories = metacat::theory::RawTheorySet::from_files(paths)?;
        Self::from_raw_theories(raw_theories, dialect)
    }

    /// Construct a new runtime from in-memory Catena source strings.
    pub fn from_sources<'a, I>(sources: I, dialect: GpuDialect) -> Result<Runtime, InitError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let raw_theories = RawTheorySet::from_texts(sources)?;
        Self::from_raw_theories(raw_theories, dialect)
    }

    fn from_raw_theories(
        raw_theories: RawTheorySet,
        dialect: GpuDialect,
    ) -> Result<Runtime, InitError> {
        let report = crate::compile::compile(raw_theories)?;
        let modules = report
            .gpu_modules
            .as_ref()
            .ok_or(InitError::MissingGpuModules)?;
        let signature_table = signatures(modules);

        let report_dir = tempfile::Builder::new()
            .prefix("catena-report-")
            .tempdir()
            .map_err(|source| InitError::CreateBuildDir {
                path: std::env::temp_dir(),
                source,
            })?;
        let cpp_path = report_dir.path().join("module.cpp");
        let rendered = render_modules(modules, dialect)
            .map_err(|source| InitError::RenderGpu { dialect, source })?;
        fs::write(&cpp_path, rendered).map_err(|source| InitError::WriteGeneratedSource {
            path: cpp_path.clone(),
            source,
        })?;
        let artifact = super::artifact::compile(&cpp_path, dialect)?;

        let library = load_generated_library(artifact.path())?;
        let executor = Executor::new(library, &signature_table).map_err(|error| match error {
            ExecutorError::LoadSymbol { symbol, source } => {
                InitError::LoadSymbol { symbol, source }
            }
        })?;
        let gpu = Arc::new(GpuRuntime::load(dialect)?);

        Ok(Self {
            _artifact: artifact,
            executor,
            gpu,
            signatures: signature_table,
        })
    }

    pub fn mem_u64(&self, values: &[u64]) -> Result<Value, MemError> {
        Mem::from_u64_slice(self.gpu.clone(), values).map(Value::Mem)
    }

    pub fn mem_f32(&self, values: &[f32]) -> Result<Value, MemError> {
        Mem::from_f32_slice(self.gpu.clone(), values).map(Value::Mem)
    }

    /// Run a source-level `program` definition, which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let signature = self
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        self.exec_symbol(name, signature, args)
    }

    fn exec_symbol<const M: usize, const N: usize>(
        &self,
        name: &str,
        signature: &FunctionSignature,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        // Check arity/coarity lines up with what's in the function signature.
        if signature.inputs.len() != M {
            return Err(ExecError::InputArityMismatch {
                name: name.to_string(),
                expected: signature.inputs.len(),
                actual: M,
            });
        }
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: name.to_string(),
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
        }

        self.executor
            .call(&signature.symbol, &args, &mut output_values);

        Ok(output_values
            .try_into()
            .expect("output arity already validated"))
    }

    fn zeroed_value(&self, kind: ValueKind) -> Value {
        match kind {
            ValueKind::Bool => Value::Bool(0),
            ValueKind::U32 => Value::U32(0),
            ValueKind::U64 => Value::U64(0),
            ValueKind::F32 => Value::F32(0.0),
            ValueKind::Mem => Value::Mem(Mem::null(self.gpu.clone())),
        }
    }
}

fn load_generated_library(path: &Path) -> Result<Library, InitError> {
    // Generated GPU shared objects must remain resident for the process lifetime.
    // If one is unloaded and a generated GPU object is loaded again later, ROCm/LLVM
    // initialization can re-register process-global LLVM command-line options and
    // abort with "Option 'ubsan-guard-checks' registered more than once".
    // RTLD_NODELETE lets the Rust handle be dropped while preventing that unload.
    let flags = RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NODELETE;
    let library = unsafe { UnixLibrary::open(Some(path), flags) }.map_err(|source| {
        InitError::LoadLibrary {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok(library.into())
}
