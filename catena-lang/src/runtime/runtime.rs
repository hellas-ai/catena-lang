use thiserror::Error;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use libloading::Library;
use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
use serde::{Deserialize, Serialize};

use super::artifact::{Artifact, ArtifactError, RuntimeId, SharedObject};
use super::executor::{AbiValue, Executor, ExecutorError};
use super::mem::{MemError, MemOwn};
use super::{
    signature::{
        FunctionSignature, GeneratedFunction, SignatureTable, generated_signatures, signatures,
    },
    value::{Value, ValueKind},
};
use crate::codegen::{GpuDialect, gpu::GpuRenderError, gpu::render_modules};
use crate::compile::CompileFailure;
use crate::gpu::GpuApi;
use metacat::theory::RawTheorySet;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    /// GPU operations used to validate and release memory crossing the ABI.
    gpu: Arc<GpuApi>,
    runtime_id: RuntimeId,
    artifacts: Vec<LoadedArtifact>,
}

#[derive(Debug)]
struct LoadedArtifact {
    // Keep the tempdir-backed shared object alive for as long as the library is loaded.
    _shared_object: SharedObject,
    /// Prepared entry points in the loaded shared object.
    executor: Executor,
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
    #[error("Function '{name}' has unsupported cap.ref output at index {index}")]
    UnsupportedRefOutput { name: String, index: usize },
    #[error(transparent)]
    Mem(#[from] MemError),
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExecError {
    #[error("Artifact does not belong to this runtime")]
    UnknownArtifact,
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
    #[error("Argument {index} contains device memory from a different GPU dialect")]
    IncompatibleDeviceMemory { index: usize },
}

impl Runtime {
    pub(crate) fn gpu_api(&self) -> Arc<GpuApi> {
        self.gpu.clone()
    }

    /// Construct an empty runtime for the selected GPU dialect.
    pub fn new(dialect: GpuDialect) -> Result<Runtime, InitError> {
        Ok(Self {
            gpu: GpuApi::load(dialect)?,
            runtime_id: RuntimeId::new(),
            artifacts: Vec::new(),
        })
    }

    /// The GPU dialect used by this runtime.
    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect()
    }

    /// Compile Catena programs from paths into a new artifact.
    pub fn load<I>(&mut self, paths: I) -> Result<Artifact, InitError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let raw_theories = metacat::theory::RawTheorySet::from_files(paths)?;
        self.load_raw_theories(raw_theories)
    }

    /// Compile in-memory Catena source strings into a new artifact.
    pub fn load_sources<'a, I>(&mut self, sources: I) -> Result<Artifact, InitError>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let raw_theories = RawTheorySet::from_texts(sources)?;
        self.load_raw_theories(raw_theories)
    }

    fn load_raw_theories(&mut self, raw_theories: RawTheorySet) -> Result<Artifact, InitError> {
        let dialect = self.gpu.dialect();
        let report = crate::compile::compile(raw_theories)?;
        let modules = report
            .gpu_modules
            .as_ref()
            .ok_or(InitError::MissingGpuModules)?;
        let rendered = render_modules(modules, dialect)
            .map_err(|source| InitError::RenderGpu { dialect, source })?;
        self.load_generated(&rendered, signatures(modules))
    }

    /// Compile and load generated GPU source with its public entry-point ABI.
    ///
    /// This lets another Catena compiler reuse the runtime without depending on
    /// catena-lang's compiler or code-generation representation.
    pub fn load_generated_source(
        &mut self,
        source: &str,
        functions: impl IntoIterator<Item = GeneratedFunction>,
    ) -> Result<Artifact, InitError> {
        self.load_generated(source, generated_signatures(functions))
    }

    fn load_generated(
        &mut self,
        source: &str,
        signature_table: SignatureTable,
    ) -> Result<Artifact, InitError> {
        let dialect = self.gpu.dialect();
        if let Some((name, index)) = ref_output(&signature_table) {
            return Err(InitError::UnsupportedRefOutput { name, index });
        }

        let report_dir = tempfile::Builder::new()
            .prefix("catena-report-")
            .tempdir()
            .map_err(|source| InitError::CreateBuildDir {
                path: std::env::temp_dir(),
                source,
            })?;
        let cpp_path = report_dir.path().join("module.cpp");
        fs::write(&cpp_path, source).map_err(|source| InitError::WriteGeneratedSource {
            path: cpp_path.clone(),
            source,
        })?;
        let shared_object = super::artifact::compile(&cpp_path, dialect)?;

        let library = load_generated_library(shared_object.path())?;
        let executor = Executor::new(library, &signature_table).map_err(|error| match error {
            ExecutorError::LoadSymbol { symbol, source } => {
                InitError::LoadSymbol { symbol, source }
            }
        })?;
        let artifact = Artifact::new(self.runtime_id, self.artifacts.len());
        self.artifacts.push(LoadedArtifact {
            _shared_object: shared_object,
            executor,
            signatures: signature_table,
        });
        Ok(artifact)
    }

    pub fn mem_u64(&self, values: &[u64]) -> Result<MemOwn, MemError> {
        MemOwn::from_u64_slice(values, self.gpu.dialect())
    }

    pub fn mem_u16(&self, values: &[u16]) -> Result<MemOwn, MemError> {
        MemOwn::from_u16_slice(values, self.gpu.dialect())
    }

    pub fn mem_f32(&self, values: &[f32]) -> Result<MemOwn, MemError> {
        MemOwn::from_f32_slice(values, self.gpu.dialect())
    }

    /// Run a source-level `program` definition from `artifact`.
    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], ExecError> {
        let loaded = self.loaded_artifact(artifact)?;
        let signature = loaded
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: name.to_string(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }

        self.exec_symbol(loaded, name, signature, args.into())
            .map(|values| values.try_into().expect("output arity already validated"))
    }

    /// Run a source-level `program` from `artifact` with dynamically sized
    /// input and output collections.
    ///
    /// This is the public execution boundary for adapters such as SafeRuntime,
    /// whose arities are known from a runtime protocol rather than const
    /// generics.
    pub fn exec_values<'a>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        let loaded = self.loaded_artifact(artifact)?;
        let signature = loaded
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        self.exec_symbol(loaded, name, signature, args)
    }

    fn exec_symbol<'a>(
        &self,
        loaded: &LoadedArtifact,
        name: &str,
        signature: &FunctionSignature,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        // Check input arity lines up with what's in the function signature.
        if signature.inputs.len() != args.len() {
            return Err(ExecError::InputArityMismatch {
                name: name.to_string(),
                expected: signature.inputs.len(),
                actual: args.len(),
            });
        }
        let mut raw_outputs = signature
            .outputs
            .iter()
            .copied()
            .map(AbiValue::zeroed)
            .collect::<Vec<_>>();

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
            let memory_dialect = match value {
                Value::MemOwn(memory) => Some(memory.dialect()),
                Value::MemRef(memory) => Some(memory.dialect()),
                _ => None,
            };
            if let Some(dialect) = memory_dialect
                && self.gpu.dialect() != dialect
            {
                return Err(ExecError::IncompatibleDeviceMemory { index });
            }
        }

        let raw_inputs = args
            .into_iter()
            .map(|value| match value {
                Value::Bool(value) => AbiValue::Bool(value),
                Value::U16(value) => AbiValue::U16(value),
                Value::U32(value) => AbiValue::U32(value),
                Value::U64(value) => AbiValue::U64(value),
                Value::F32(value) => AbiValue::F32(value),
                Value::MemOwn(memory) => AbiValue::Mem(memory.into_abi()),
                Value::MemRef(memory) => AbiValue::Mem(memory.abi),
            })
            .collect::<Vec<_>>();

        loaded
            .executor
            .call(&signature.symbol, &raw_inputs, &mut raw_outputs);

        raw_outputs
            .into_iter()
            .map(|output| self.resolve_output(output))
            .collect()
    }

    fn resolve_output(&self, output: AbiValue) -> Result<Value<'static>, ExecError> {
        match output {
            AbiValue::Bool(value) => Ok(Value::Bool(value)),
            AbiValue::U16(value) => Ok(Value::U16(value)),
            AbiValue::U32(value) => Ok(Value::U32(value)),
            AbiValue::U64(value) => Ok(Value::U64(value)),
            AbiValue::F32(value) => Ok(Value::F32(value)),
            AbiValue::Mem(abi) => {
                // SAFETY: cap.ref outputs are rejected at initialization, so
                // every memory output transfers a GPU allocation owned by the
                // generated program to its caller.
                let memory =
                    unsafe { MemOwn::from_raw_parts_with_gpu(abi.data, abi.len, self.gpu.clone()) };
                Ok(Value::from(memory))
            }
        }
    }

    fn loaded_artifact(&self, artifact: &Artifact) -> Result<&LoadedArtifact, ExecError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(ExecError::UnknownArtifact);
        }
        self.artifacts
            .get(artifact.index())
            .ok_or(ExecError::UnknownArtifact)
    }

    pub(crate) fn artifact_at(&self, index: usize) -> Result<Artifact, ExecError> {
        self.artifacts
            .get(index)
            .ok_or(ExecError::UnknownArtifact)?;
        Ok(Artifact::new(self.runtime_id, index))
    }
}

fn ref_output(signatures: &SignatureTable) -> Option<(String, usize)> {
    signatures.iter().find_map(|(name, signature)| {
        signature
            .outputs
            .iter()
            .position(|kind| *kind == ValueKind::MemRef)
            .map(|index| (name.clone(), index))
    })
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
