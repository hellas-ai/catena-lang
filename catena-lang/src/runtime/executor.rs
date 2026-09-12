//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::{collections::HashMap, ffi::c_void};

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

use crate::codegen::{
    GENERATED_ALLOCATION_BUDGET_BEGIN_SYMBOL, GENERATED_ALLOCATION_BUDGET_END_SYMBOL,
};

use super::{signature::SignatureTable, value::ValueKind};

type AllocationBudgetBegin = unsafe extern "C" fn(u64);
type AllocationBudgetEnd = unsafe extern "C" fn();

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct CatenaMem {
    pub(super) data: *mut c_void,
    pub(super) len: u64,
}

/// Temporary storage for values crossing the generated C ABI.
///
/// Unlike [`super::value::Value`], this type does not express Rust ownership:
/// an owned input has already relinquished its allocation before becoming an
/// `AbiValue`, while a memory output is not wrapped in an owning Rust value
/// until generated code has filled its [`CatenaMem`] slot. It must therefore
/// remain private to the synchronous executor boundary.
#[derive(Debug)]
pub(super) enum AbiValue {
    Bool(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    Mem(CatenaMem),
}

impl AbiValue {
    pub(super) fn zeroed(kind: ValueKind) -> Self {
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

#[derive(Debug, Clone, Copy)]
struct AllocationBudgetControl {
    begin: AllocationBudgetBegin,
    end: AllocationBudgetEnd,
}

impl AllocationBudgetControl {
    fn under_limit<T>(&self, byte_limit: u64, invoke: impl FnOnce() -> T) -> T {
        unsafe { (self.begin)(byte_limit) };
        let _reset = AllocationBudgetReset { control: self };
        invoke()
    }
}

struct AllocationBudgetReset<'a> {
    control: &'a AllocationBudgetControl,
}

impl Drop for AllocationBudgetReset<'_> {
    fn drop(&mut self) {
        unsafe { (self.control.end)() };
    }
}

/// Loaded generated code and its prepared dynamic call interfaces.
#[derive(Debug)]
pub(super) struct Executor {
    // Keep the library loaded for as long as any cached code pointer can be used.
    _library: Library,
    functions: HashMap<String, PreparedFunction>,
    allocation_budget: Option<AllocationBudgetControl>,
}

#[derive(Debug, Error)]
pub(super) enum ExecutorError {
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
}

impl Executor {
    /// Resolve generated entry points and prepare their libffi call interfaces once.
    pub(super) fn new(
        library: Library,
        signatures: &SignatureTable,
    ) -> Result<Self, ExecutorError> {
        let allocation_budget = AllocationBudgetControl {
            begin: load_control_symbol(&library, GENERATED_ALLOCATION_BUDGET_BEGIN_SYMBOL)?,
            end: load_control_symbol(&library, GENERATED_ALLOCATION_BUDGET_END_SYMBOL)?,
        };
        Self::with_allocation_budget(library, signatures, Some(allocation_budget))
    }

    /// External compiler output need not implement Catena's allocation controls.
    /// This constructor never enables execution with an allocation budget.
    #[cfg(feature = "experimental-catena-gpu")]
    pub(super) fn external(
        library: Library,
        signatures: &SignatureTable,
    ) -> Result<Self, ExecutorError> {
        Self::with_allocation_budget(library, signatures, None)
    }

    fn with_allocation_budget(
        library: Library,
        signatures: &SignatureTable,
        allocation_budget: Option<AllocationBudgetControl>,
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
            allocation_budget,
        })
    }

    /// Invoke a prepared symbol. Runtime validation guarantees the value shapes and kinds.
    pub(super) fn call(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
        match self.allocation_budget {
            Some(control) => {
                control.under_limit(u64::MAX, || self.call_raw(symbol, inputs, outputs))
            }
            None => self.call_raw(symbol, inputs, outputs),
        }
    }

    /// Invoke a prepared symbol with a cumulative generated-device-allocation limit.
    pub(super) fn call_with_generated_allocation_budget(
        &self,
        symbol: &str,
        inputs: &[AbiValue],
        outputs: &mut [AbiValue],
        byte_limit: u64,
    ) {
        self.allocation_budget
            .expect("budget support is checked before input ownership is transferred")
            .under_limit(byte_limit, || self.call_raw(symbol, inputs, outputs));
    }

    pub(super) fn supports_allocation_budget(&self) -> bool {
        self.allocation_budget.is_some()
    }

    fn call_raw(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
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

        unsafe { function.cif.call::<()>(function.code, &arguments) };
    }
}

fn load_control_symbol<T: Copy>(library: &Library, symbol: &str) -> Result<T, ExecutorError> {
    let symbol_name = format!("{symbol}\0");
    unsafe { library.get::<T>(symbol_name.as_bytes()) }
        .map(|loaded| *loaded)
        .map_err(|source| ExecutorError::LoadSymbol {
            symbol: symbol.to_string(),
            source,
        })
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static ACTIVE_LIMIT: AtomicU64 = AtomicU64::new(u64::MAX);
    static END_COUNT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" fn begin(byte_limit: u64) {
        ACTIVE_LIMIT.store(byte_limit, Ordering::SeqCst);
    }

    unsafe extern "C" fn end() {
        ACTIVE_LIMIT.store(u64::MAX, Ordering::SeqCst);
        END_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn normal_loader_requires_allocation_control_symbols() {
        let library = libloading::os::unix::Library::this().into();
        assert!(matches!(
            Executor::new(library, &SignatureTable::new()),
            Err(ExecutorError::LoadSymbol { symbol, .. })
                if symbol == GENERATED_ALLOCATION_BUDGET_BEGIN_SYMBOL
        ));
    }

    #[cfg(feature = "experimental-catena-gpu")]
    #[test]
    fn external_executor_runs_without_allocation_control_symbols() {
        unsafe extern "C" fn write_output(output: *mut u64) {
            unsafe { output.write(42) };
        }
        let library = libloading::os::unix::Library::this().into();
        let mut executor = Executor::external(library, &SignatureTable::new()).unwrap();
        assert!(!executor.supports_allocation_budget());
        executor.functions.insert(
            "fixture".into(),
            PreparedFunction {
                code: CodePtr(write_output as *mut c_void),
                cif: Cif::new([Type::pointer()], Type::void()),
            },
        );
        let mut outputs = [AbiValue::U64(0)];
        executor.call("fixture", &[], &mut outputs);
        assert!(matches!(outputs, [AbiValue::U64(42)]));
    }

    #[test]
    fn allocation_budget_is_reset_between_invocations_and_on_unwind() {
        let control = AllocationBudgetControl { begin, end };
        END_COUNT.store(0, Ordering::SeqCst);

        control.under_limit(17, || {
            assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), 17);
        });
        assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), u64::MAX);

        let panic = std::panic::catch_unwind(|| {
            control.under_limit(23, || panic!("fixture panic"));
        });
        assert!(panic.is_err());
        assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), u64::MAX);
        assert_eq!(END_COUNT.load(Ordering::SeqCst), 2);
    }
}
