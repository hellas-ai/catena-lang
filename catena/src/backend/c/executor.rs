//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::ffi::c_void;
use std::path::Path;

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

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
    #[error("Failed to load shared object: {0}")]
    LoadLibrary(#[source] libloading::Error),
    #[error("Failed to resolve symbol '{symbol}': {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
}

/// Invoke a compiled symbol using the generated C ABI.
///
/// The executor only knows about ABI-level scalar slots and output pointers.
/// Catena-specific type mapping belongs in `runtime`.
pub(crate) fn exec(
    so_path: &Path,
    symbol: &str,
    frame: CallFrame<'_>,
) -> Result<(), ExecutorError> {
    let library = unsafe { Library::new(so_path) }.map_err(ExecutorError::LoadLibrary)?;
    let symbol_name = format!("{symbol}\0");
    let function =
        unsafe { library.get::<*mut c_void>(symbol_name.as_bytes()) }.map_err(|source| {
            ExecutorError::LoadSymbol {
                symbol: symbol.to_string(),
                source,
            }
        })?;

    let mut pointer_args = Vec::new();
    let mut types = Vec::with_capacity(frame.args.len());
    for arg in frame.args.iter_mut() {
        match arg {
            ArgValue::U64(_) => types.push(Type::u64()),
            ArgValue::F32(_) => types.push(Type::f32()),
            ArgValue::OutU64(slot) => {
                types.push(Type::pointer());
                pointer_args.push((*slot as *mut u64).cast::<c_void>());
            }
            ArgValue::OutF32(slot) => {
                types.push(Type::pointer());
                pointer_args.push((*slot as *mut f32).cast::<c_void>());
            }
        }
    }

    let cif = Cif::new(types, Type::void());
    let mut pointer_index = 0usize;
    let args: Vec<Arg<'_>> = frame
        .args
        .iter()
        .map(|arg| match arg {
            ArgValue::U64(value) => Arg::new(*value),
            ArgValue::F32(value) => Arg::new(*value),
            ArgValue::OutU64(_) | ArgValue::OutF32(_) => {
                let ptr = &pointer_args[pointer_index];
                pointer_index += 1;
                Arg::new(ptr)
            }
        })
        .collect();

    unsafe {
        cif.call::<()>(CodePtr(*function), &args);
    }
    Ok(())
}
