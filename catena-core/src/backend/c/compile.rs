//! Compile a catena program to a C shared object file.
use crate::backend::c::codegen::codegen;
use crate::backend::c::value::ValueKind;
use crate::lang::Obj;
use crate::lower::{LowerError, Pass, lower};
use metacat::theory::Theory;
use metacat::tree::Tree;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Failed to parse program: {0}")]
    Parse(#[from] metacat::theory::load::LoadError),
    #[error("No def-arrow declarations found to compile")]
    NoDefinitions,
    #[error("Failed to lower definition '{definition}': {source}")]
    Lower {
        definition: String,
        source: LowerError,
    },
    #[error("Function '{definition}' uses an unsupported runtime value type: {value:?}")]
    UnsupportedRuntimeType { definition: String, value: Obj },
    #[error("Failed to create temporary build directory: {0}")]
    TempDir(#[from] std::io::Error),
    #[error("C compiler is unavailable: {0}")]
    CompilerUnavailable(std::io::Error),
    #[error("C compilation failed with status {status}: {stderr}")]
    CompilerFailed { status: ExitStatus, stderr: String },
}

/// A shared object file created by compiling a catena program
#[derive(Debug)]
pub(crate) struct SharedObject {
    _build_dir: tempfile::TempDir,
    path: PathBuf,
    signatures: HashMap<String, FunctionSignature>,
}

impl SharedObject {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn signature(&self, definition: &str) -> Option<&FunctionSignature> {
        self.signatures.get(definition)
    }
}

/// FunctionSignature represents the input and output types to each C function.
#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub(crate) symbol: String,
    pub(crate) inputs: Vec<ValueKind>,
    pub(crate) outputs: Vec<ValueKind>,
}

pub(crate) fn compile(theory: &Theory) -> Result<SharedObject, CompileError> {
    let mut definitions: Vec<String> = match theory {
        Theory::Nat => Vec::new(),
        Theory::Theory { arrows, .. } => arrows
            .iter()
            .filter(|(_, arrow)| arrow.definition.is_some())
            .map(|(definition, _)| definition.as_str().to_string())
            .collect(),
    };
    definitions.sort();
    if definitions.is_empty() {
        return Err(CompileError::NoDefinitions);
    }

    // Compile each definition to C, and record its function signature
    let mut translation_unit = String::from("#include <stdint.h>\n\n");
    let mut used_symbols = HashSet::new();
    let mut signatures = HashMap::new();
    for definition in definitions {
        let lowered = lower(theory, Pass::DiscardNaturality, &definition).map_err(|source| {
            CompileError::Lower {
                definition: definition.clone(),
                source,
            }
        })?;

        let symbol = unique_symbol(&definition, &mut used_symbols);
        let signature = FunctionSignature {
            symbol: symbol.clone(),
            inputs: value_kinds(&definition, &lowered.sources, &lowered.hypergraph.nodes)?,
            outputs: value_kinds(&definition, &lowered.targets, &lowered.hypergraph.nodes)?,
        };
        let function = codegen(lowered, &symbol);

        translation_unit.push_str(&function);
        translation_unit.push_str("\n\n");
        signatures.insert(definition, signature);
    }

    // Set up temp file dirs
    let build_dir = tempfile::Builder::new().prefix("catena-c-").tempdir()?;
    let c_path = build_dir.path().join("module.c");
    let so_path = build_dir.path().join("module.so");
    std::fs::write(&c_path, translation_unit)?;

    // Compile the generated C to a shared object
    let output = Command::new("cc")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O2")
        .arg(&c_path)
        .arg("-o")
        .arg(&so_path)
        .output()
        .map_err(CompileError::CompilerUnavailable)?;

    if !output.status.success() {
        return Err(CompileError::CompilerFailed {
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    Ok(SharedObject {
        _build_dir: build_dir,
        path: so_path,
        signatures,
    })
}

////////////////////////////////////////////////////////////////////////////////
// Symbol mangling

fn unique_symbol(definition: &str, used: &mut HashSet<String>) -> String {
    let base = mangle_symbol(definition);
    if used.insert(base.clone()) {
        return base;
    }

    let mut i = 1usize;
    loop {
        let symbol = format!("{base}_{i}");
        if used.insert(symbol.clone()) {
            return symbol;
        }
        i += 1;
    }
}

fn mangle_symbol(name: &str) -> String {
    let mut symbol = String::new();
    for (index, ch) in name.chars().enumerate() {
        let valid = ch.is_ascii_alphanumeric() || ch == '_';
        if valid {
            if index == 0 && ch.is_ascii_digit() {
                symbol.push('_');
            }
            symbol.push(ch);
        } else {
            symbol.push('_');
        }
    }
    if symbol.is_empty() {
        "_".to_string()
    } else {
        symbol
    }
}

////////////////////////////////////////////////////////////////////////////////
// Type -> ValueKind

fn value_kinds(
    definition: &str,
    interface: &[open_hypergraphs::lax::NodeId],
    nodes: &[Obj],
) -> Result<Vec<ValueKind>, CompileError> {
    interface
        .iter()
        .map(|node_id| value_kind(definition, &nodes[node_id.0]))
        .collect()
}

fn value_kind(definition: &str, obj: &Obj) -> Result<ValueKind, CompileError> {
    match obj {
        Tree::Node(val, 0, children) if val.as_str() == "value" => {
            let [Tree::Node(key, 0, _)] = children.as_slice() else {
                return Err(CompileError::UnsupportedRuntimeType {
                    definition: definition.to_string(),
                    value: obj.clone(),
                });
            };
            match key.as_str() {
                "f32" => Ok(ValueKind::F32),
                "index" => Ok(ValueKind::Index),
                "extent" => Ok(ValueKind::Extent),
                _ => Err(CompileError::UnsupportedRuntimeType {
                    definition: definition.to_string(),
                    value: obj.clone(),
                }),
            }
        }
        _ => Err(CompileError::UnsupportedRuntimeType {
            definition: definition.to_string(),
            value: obj.clone(),
        }),
    }
}
