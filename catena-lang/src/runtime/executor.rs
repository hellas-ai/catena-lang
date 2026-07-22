//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::{collections::HashMap, ffi::c_void};

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

use super::{
    signature::SignatureTable,
    value::{Value, ValueKind},
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CatenaMem {
    pub(crate) data: *mut c_void,
    pub(crate) len: u64,
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
    pub(crate) fn call(&self, symbol: &str, inputs: &[Value], outputs: &mut [Value]) {
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
        ValueKind::U32 => Type::u32(),
        ValueKind::U64 => Type::u64(),
        ValueKind::F32 => Type::f32(),
        ValueKind::Mem => Type::structure([Type::pointer(), Type::u64()]),
    }
}

fn input_arg(value: &Value) -> Arg<'_> {
    match value {
        Value::Bool(value) => Arg::new(value),
        Value::U32(value) => Arg::new(value),
        Value::U64(value) => Arg::new(value),
        Value::F32(value) => Arg::new(value),
        Value::Mem(value) => Arg::new(&value.abi),
    }
}

fn output_pointer(value: &mut Value) -> *mut c_void {
    match value {
        Value::Bool(value) => (value as *mut u8).cast(),
        Value::U32(value) => (value as *mut u32).cast(),
        Value::U64(value) => (value as *mut u64).cast(),
        Value::F32(value) => (value as *mut f32).cast(),
        Value::Mem(value) => (&mut value.abi as *mut CatenaMem).cast(),
    }
}
