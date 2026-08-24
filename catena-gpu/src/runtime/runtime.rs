use std::fs;

use libloading::{
    Library,
    os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL},
};
use metacat::theory::RawTheorySet;
use thiserror::Error;

use crate::{
    codegen::{
        GpuDialect,
        gpu::{GpuRenderError, render_modules},
    },
    compile::CompileFailure,
};

use super::{
    Artifact, ArtifactError, Value, ValueKind,
    artifact::{RuntimeId, SharedObject, compile as compile_artifact},
    executor::{AbiValue, Executor, ExecutorError},
    signature::{FunctionSignature, SignatureTable, signatures},
};

#[derive(Debug)]
pub struct Runtime {
    dialect: GpuDialect,
    id: RuntimeId,
    artifacts: Vec<LoadedArtifact>,
}

#[derive(Debug)]
struct LoadedArtifact {
    _object: SharedObject,
    executor: Executor,
    signatures: SignatureTable,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error(transparent)]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error(transparent)]
    Compile(#[from] CompileFailure),
    #[error("compile report did not contain GPU modules")]
    MissingModules,
    #[error(transparent)]
    Render(#[from] GpuRenderError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("failed to load generated library: {0}")]
    LoadLibrary(#[from] libloading::Error),
    #[error("failed to load generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        source: libloading::Error,
    },
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("artifact does not belong to this runtime")]
    UnknownArtifact,
    #[error("unknown source function `{0}`")]
    UnknownFunction(String),
    #[error("function `{name}` expected {expected} inputs, got {actual}")]
    InputArity {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("function `{name}` expected {expected} outputs, got {actual}")]
    OutputArity {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("argument {index} expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        index: usize,
        expected: ValueKind,
        actual: ValueKind,
    },
}

impl Runtime {
    pub fn new(dialect: GpuDialect) -> Result<Self, InitError> {
        Ok(Self {
            dialect,
            id: RuntimeId::new(),
            artifacts: vec![],
        })
    }

    pub fn load_sources<'a>(
        &mut self,
        sources: impl IntoIterator<Item = &'a str>,
    ) -> Result<Artifact, InitError> {
        let raw = RawTheorySet::from_texts(sources)?;
        let report = crate::compile::compile(raw)?;
        let modules = report
            .gpu_modules
            .as_ref()
            .ok_or(InitError::MissingModules)?;
        let signatures = signatures(modules);
        let directory = tempfile::Builder::new()
            .prefix("catena-gpu-source-")
            .tempdir()?;
        let source_path = directory.path().join("module.cpp");
        fs::write(&source_path, render_modules(modules, self.dialect)?)?;
        let object = compile_artifact(&source_path, self.dialect)?;
        let library = load_library(object.path())?;
        let executor = Executor::new(library, &signatures).map_err(|error| match error {
            ExecutorError::LoadSymbol { symbol, source } => {
                InitError::LoadSymbol { symbol, source }
            }
        })?;
        let artifact = Artifact {
            runtime: self.id,
            index: self.artifacts.len(),
        };
        self.artifacts.push(LoadedArtifact {
            _object: object,
            executor,
            signatures,
        });
        Ok(artifact)
    }

    pub fn exec<const M: usize, const N: usize>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let loaded = self.loaded(artifact)?;
        let signature = loaded
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownFunction(name.into()))?;
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArity {
                name: name.into(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }
        let values = execute(loaded, name, signature, args.into())?;
        Ok(values.try_into().expect("output arity was checked"))
    }

    fn loaded(&self, artifact: &Artifact) -> Result<&LoadedArtifact, ExecError> {
        if artifact.runtime != self.id {
            return Err(ExecError::UnknownArtifact);
        }
        self.artifacts
            .get(artifact.index)
            .ok_or(ExecError::UnknownArtifact)
    }
}

fn execute(
    loaded: &LoadedArtifact,
    name: &str,
    signature: &FunctionSignature,
    args: Vec<Value>,
) -> Result<Vec<Value>, ExecError> {
    if signature.inputs.len() != args.len() {
        return Err(ExecError::InputArity {
            name: name.into(),
            expected: signature.inputs.len(),
            actual: args.len(),
        });
    }
    for (index, (actual, expected)) in args.iter().zip(&signature.inputs).enumerate() {
        if actual.kind() != *expected {
            return Err(ExecError::TypeMismatch {
                index,
                expected: *expected,
                actual: actual.kind(),
            });
        }
    }
    let inputs = args.into_iter().map(AbiValue::from).collect::<Vec<_>>();
    let mut outputs = signature
        .outputs
        .iter()
        .copied()
        .map(AbiValue::zeroed)
        .collect::<Vec<_>>();
    loaded
        .executor
        .call(&signature.symbol, &inputs, &mut outputs);
    Ok(outputs.into_iter().map(Value::from).collect())
}

fn load_library(path: &std::path::Path) -> Result<Library, libloading::Error> {
    let library = unsafe { UnixLibrary::open(Some(path), RTLD_LAZY | RTLD_LOCAL)? };
    Ok(library.into())
}

impl From<std::io::Error> for InitError {
    fn from(error: std::io::Error) -> Self {
        Self::Artifact(ArtifactError::Io(error))
    }
}
