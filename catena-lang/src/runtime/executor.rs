//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::{collections::HashMap, ffi::c_void};

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

use super::{signature::SignatureTable, value::ValueKind};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CatenaMem {
    pub(crate) data: *mut c_void,
    pub(crate) len: u64,
}

/// Temporary storage for values crossing the generated C ABI.
///
/// Unlike [`super::value::Value`], this type does not express Rust ownership:
/// an owned input has already relinquished its allocation before becoming an
/// `AbiValue`, while a memory output is not wrapped in an owning Rust value
/// until generated code has filled its [`CatenaMem`] slot. It must therefore
/// remain private to the synchronous executor boundary.
#[derive(Debug)]
pub(crate) enum AbiValue {
    Bool(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    Mem(CatenaMem),
}

impl AbiValue {
    pub(crate) fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Bool => Self::Bool(0),
            ValueKind::U16 => Self::U16(0),
            ValueKind::U32 => Self::U32(0),
            ValueKind::U64 => Self::U64(0),
            ValueKind::F32 => Self::F32(0.0),
            ValueKind::MemOwn | ValueKind::MemRef => Self::Mem(CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            }),
        }
    }
}

#[derive(Debug)]
struct PreparedFunction {
    code: CodePtr,
    cif: Cif,
}

/// Loaded generated code and its prepared dynamic call interfaces.
#[derive(Debug)]
pub(crate) struct Executor {
    // Keep the library loaded for as long as any cached code pointer can be used.
    _library: Library,
    functions: HashMap<String, PreparedFunction>,
}

#[derive(Debug, Error)]
pub(crate) enum ExecutorError {
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
}

impl Executor {
    /// Resolve generated entry points and prepare their libffi call interfaces once.
    pub(crate) fn new(
        library: Library,
        signatures: &SignatureTable,
    ) -> Result<Self, ExecutorError> {
        let mut functions = HashMap::with_capacity(signatures.len());
        for signature in signatures.values() {
            if functions.contains_key(&signature.symbol) {
                continue;
            }

            let symbol_name = format!("{}\0", signature.symbol);
            let function = unsafe { library.get::<*mut c_void>(symbol_name.as_bytes()) }.map_err(
                |source| ExecutorError::LoadSymbol {
                    symbol: signature.symbol.clone(),
                    source,
                },
            )?;

            let argument_types = signature
                .inputs
                .iter()
                .copied()
                .map(ffi_type)
                .chain(signature.outputs.iter().map(|_| Type::pointer()))
                .collect::<Vec<_>>();
            functions.insert(
                signature.symbol.clone(),
                PreparedFunction {
                    code: CodePtr(*function),
                    cif: Cif::new(argument_types, Type::void()),
                },
            );
        }

        Ok(Self {
            _library: library,
            functions,
        })
    }

    /// Invoke a prepared symbol. Runtime validation guarantees the value shapes and kinds.
    pub(crate) fn call(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
        let function = self
            .functions
            .get(symbol)
            .expect("source signature should have a prepared generated function");

        // A generated output parameter is itself a pointer value. libffi therefore needs
        // an address containing that pointer while it constructs the native call frame.
        let output_pointers = outputs.iter_mut().map(output_pointer).collect::<Vec<_>>();
        let arguments = inputs
            .iter()
            .map(input_arg)
            .chain(output_pointers.iter().map(Arg::new))
            .collect::<Vec<_>>();

        unsafe {
            function.cif.call::<()>(function.code, &arguments);
        }
    }
}

fn ffi_type(kind: ValueKind) -> Type {
    match kind {
        ValueKind::Bool => Type::u8(),
        ValueKind::U16 => Type::u16(),
        ValueKind::U32 => Type::u32(),
        ValueKind::U64 => Type::u64(),
        ValueKind::F32 => Type::f32(),
        ValueKind::MemOwn | ValueKind::MemRef => Type::structure([Type::pointer(), Type::u64()]),
    }
}

fn input_arg(value: &AbiValue) -> Arg<'_> {
    match value {
        AbiValue::Bool(value) => Arg::new(value),
        AbiValue::U16(value) => Arg::new(value),
        AbiValue::U32(value) => Arg::new(value),
        AbiValue::U64(value) => Arg::new(value),
        AbiValue::F32(value) => Arg::new(value),
        AbiValue::Mem(value) => Arg::new(value),
    }
}

fn output_pointer(value: &mut AbiValue) -> *mut c_void {
    match value {
        AbiValue::Bool(value) => (value as *mut u8).cast(),
        AbiValue::U16(value) => (value as *mut u16).cast(),
        AbiValue::U32(value) => (value as *mut u32).cast(),
        AbiValue::U64(value) => (value as *mut u64).cast(),
        AbiValue::F32(value) => (value as *mut f32).cast(),
        AbiValue::Mem(value) => (value as *mut CatenaMem).cast(),
    }
}
