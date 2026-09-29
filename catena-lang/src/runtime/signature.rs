use std::collections::HashMap;

use crate::runtime::{GeneratedFunction, ValueKind};

#[derive(Debug, Clone)]
pub(super) struct FunctionSignature {
    pub(super) symbol: String,
    pub(super) inputs: Vec<ValueKind>,
    pub(super) outputs: Vec<ValueKind>,
}

/// Source-level program names and their generated C ABI signatures.
pub(super) type SignatureTable = HashMap<String, FunctionSignature>;

pub(super) fn generated_signatures(
    functions: impl IntoIterator<Item = GeneratedFunction>,
) -> SignatureTable {
    functions
        .into_iter()
        .map(|function| {
            (
                function.source_name,
                FunctionSignature {
                    symbol: function.symbol,
                    inputs: function.inputs,
                    outputs: function.outputs,
                },
            )
        })
        .collect()
}
