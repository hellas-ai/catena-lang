//! Compile a catena program to a C shared object file.
use crate::backend::c::codegen::codegen;
use crate::lower::{LowerError, Pass, lower};
use metacat::syntax::TheoryBundle;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Failed to parse program: {0}")]
    Parse(#[from] metacat::syntax::LoadError),
    #[error("No def-arrow declarations found to compile")]
    NoDefinitions,
    #[error("Failed to lower definition '{definition}': {source}")]
    Lower {
        definition: String,
        source: LowerError,
    },
    #[error("Failed to create temporary build directory: {0}")]
    TempDir(#[from] std::io::Error),
    #[error("C compiler is unavailable: {0}")]
    CompilerUnavailable(std::io::Error),
    #[error("C compilation failed with status {status}: {stderr}")]
    CompilerFailed { status: ExitStatus, stderr: String },
}

#[derive(Debug)]
pub(crate) struct SharedObject {
    _build_dir: tempfile::TempDir,
    path: PathBuf,
    symbols: HashMap<String, String>,
}

impl SharedObject {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn symbol(&self, definition: &str) -> Option<&str> {
        self.symbols.get(definition).map(String::as_str)
    }
}

pub(crate) fn compile(source: &str) -> Result<SharedObject, CompileError> {
    let bundle = TheoryBundle::from_text(source)?;

    let mut definitions: Vec<String> = bundle
        .definitions
        .keys()
        .map(|definition| definition.as_str().to_string())
        .collect();
    definitions.sort();
    if definitions.is_empty() {
        return Err(CompileError::NoDefinitions);
    }

    let mut translation_unit = String::from("#include <stdint.h>\n\n");
    let mut used_symbols = HashSet::new();
    let mut symbols = HashMap::new();
    for definition in definitions {
        let lowered = lower(&bundle, Pass::DiscardNaturality, &definition).map_err(|source| {
            CompileError::Lower {
                definition: definition.clone(),
                source,
            }
        })?;

        let symbol = unique_symbol(&definition, &mut used_symbols);
        let function = codegen(lowered, &symbol);

        translation_unit.push_str(&function);
        translation_unit.push_str("\n\n");
        symbols.insert(definition, symbol);
    }

    let build_dir = tempfile::Builder::new().prefix("catena-c-").tempdir()?;
    let c_path = build_dir.path().join("module.c");
    let so_path = build_dir.path().join("module.so");
    std::fs::write(&c_path, translation_unit)?;

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
        symbols,
    })
}

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
