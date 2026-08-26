use std::{collections::HashMap, ffi::c_void};

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

use super::{Value, ValueKind, signature::SignatureTable};

#[derive(Debug)]
pub(crate) enum AbiValue {
    Bool(u8),
    U32(u32),
    U64(u64),
    F32(f32),
}

impl AbiValue {
    pub fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Bool => Self::Bool(0),
            ValueKind::U32 => Self::U32(0),
            ValueKind::U64 => Self::U64(0),
            ValueKind::F32 => Self::F32(0.0),
        }
    }
}

impl From<Value> for AbiValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Bool(v) => Self::Bool(v),
            Value::U32(v) => Self::U32(v),
            Value::U64(v) => Self::U64(v),
            Value::F32(v) => Self::F32(v),
        }
    }
}

impl From<AbiValue> for Value {
    fn from(value: AbiValue) -> Self {
        match value {
            AbiValue::Bool(v) => Self::Bool(v),
            AbiValue::U32(v) => Self::U32(v),
            AbiValue::U64(v) => Self::U64(v),
            AbiValue::F32(v) => Self::F32(v),
        }
    }
}

#[derive(Debug)]
struct Prepared {
    code: CodePtr,
    cif: Cif,
}

#[derive(Debug)]
pub(crate) struct Executor {
    _library: Library,
    functions: HashMap<String, Prepared>,
}

#[derive(Debug, Error)]
pub(crate) enum ExecutorError {
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        source: libloading::Error,
    },
}

impl Executor {
    pub fn new(library: Library, signatures: &SignatureTable) -> Result<Self, ExecutorError> {
        let mut functions = HashMap::new();
        for signature in signatures.values() {
            let name = format!("{}\0", signature.symbol);
            let function =
                unsafe { library.get::<*mut c_void>(name.as_bytes()) }.map_err(|source| {
                    ExecutorError::LoadSymbol {
                        symbol: signature.symbol.clone(),
                        source,
                    }
                })?;
            let args = signature
                .inputs
                .iter()
                .copied()
                .map(ffi_type)
                .chain(signature.outputs.iter().map(|_| Type::pointer()))
                .collect::<Vec<_>>();
            functions.insert(
                signature.symbol.clone(),
                Prepared {
                    code: CodePtr(*function),
                    cif: Cif::new(args, Type::void()),
                },
            );
        }
        Ok(Self {
            _library: library,
            functions,
        })
    }

    pub fn call(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
        let function = &self.functions[symbol];
        let pointers = outputs.iter_mut().map(output_pointer).collect::<Vec<_>>();
        let args = inputs
            .iter()
            .map(input_arg)
            .chain(pointers.iter().map(Arg::new))
            .collect::<Vec<_>>();
        unsafe {
            function.cif.call::<()>(function.code, &args);
        }
    }
}

fn ffi_type(kind: ValueKind) -> Type {
    match kind {
        ValueKind::Bool => Type::u8(),
        ValueKind::U32 => Type::u32(),
        ValueKind::U64 => Type::u64(),
        ValueKind::F32 => Type::f32(),
    }
}

fn input_arg(value: &AbiValue) -> Arg<'_> {
    match value {
        AbiValue::Bool(v) => Arg::new(v),
        AbiValue::U32(v) => Arg::new(v),
        AbiValue::U64(v) => Arg::new(v),
        AbiValue::F32(v) => Arg::new(v),
    }
}

fn output_pointer(value: &mut AbiValue) -> *mut c_void {
    match value {
        AbiValue::Bool(v) => (v as *mut u8).cast(),
        AbiValue::U32(v) => (v as *mut u32).cast(),
        AbiValue::U64(v) => (v as *mut u64).cast(),
        AbiValue::F32(v) => (v as *mut f32).cast(),
    }
}
