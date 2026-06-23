use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::codegen::{GpuValue, GpuVar, runtime_type};

#[derive(Debug, Default)]
pub(crate) struct ProductBindings {
    values: HashMap<NodeId, GpuValue>,
}

#[derive(Debug, Error)]
pub enum ProductError {
    #[error("prod.intro expected exactly one output, found {actual}")]
    IntroOutputCount { actual: usize },
    #[error("product aliasing expected {expected} outputs, found {actual}")]
    AliasArityMismatch { expected: usize, actual: usize },
    #[error("prod.elim expected a product input, found `{0:?}`")]
    ElimNonProduct(GpuValue),
    #[error("prod.elim expected {expected} outputs for product input, found {actual}")]
    ElimArityMismatch { expected: usize, actual: usize },
}

impl ProductBindings {
    pub(crate) fn get(&self, node: NodeId) -> Option<GpuValue> {
        self.values.get(&node).cloned()
    }

    pub(crate) fn bind_intro(&mut self, output: &GpuVar, inputs: Vec<GpuValue>) {
        self.values.insert(output.node, GpuValue::Product(inputs));
    }

    pub(crate) fn bind_componentwise(
        &mut self,
        outputs: &[GpuVar],
        inputs: Vec<GpuValue>,
    ) -> Result<(), ProductError> {
        if inputs.len() != outputs.len() {
            return Err(ProductError::AliasArityMismatch {
                expected: inputs.len(),
                actual: outputs.len(),
            });
        }
        for (output, input) in outputs.iter().zip(inputs.into_iter()) {
            self.values.insert(output.node, input);
        }
        Ok(())
    }

    pub(crate) fn bind_node(&mut self, node: NodeId, value: GpuValue) {
        self.values.insert(node, value);
    }

    pub(crate) fn bind_elim(
        &mut self,
        inputs: &[GpuValue],
        outputs: &[GpuVar],
    ) -> Result<(), ProductError> {
        let [input] = inputs else {
            return self.bind_componentwise(outputs, inputs.to_vec());
        };
        let GpuValue::Product(items) = input else {
            return Err(ProductError::ElimNonProduct(input.clone()));
        };
        if items.len() != outputs.len() {
            return Err(ProductError::ElimArityMismatch {
                expected: items.len(),
                actual: outputs.len(),
            });
        }
        for (output, item) in outputs.iter().zip(items.iter()) {
            self.values.insert(output.node, item.clone());
        }
        Ok(())
    }
}

pub(crate) fn flatten_value<'a>(value: &'a GpuValue, out: &mut Vec<&'a GpuValue>) {
    match value {
        GpuValue::Product(items) => {
            for item in items {
                flatten_value(item, out);
            }
        }
        other => out.push(other),
    }
}

pub(crate) fn flattened_values<'a>(
    values: impl IntoIterator<Item = &'a GpuValue>,
) -> Vec<&'a GpuValue> {
    let mut out = Vec::new();
    for value in values {
        flatten_value(value, &mut out);
    }
    out
}

pub(crate) fn runtime_vars<'a>(values: impl IntoIterator<Item = &'a GpuValue>) -> Vec<&'a GpuVar> {
    flattened_values(values)
        .into_iter()
        .filter_map(|value| match value {
            GpuValue::Var(var) if runtime_type(var).is_some() => Some(var),
            _ => None,
        })
        .collect()
}

pub(crate) fn runtime_values<'a>(
    values: impl IntoIterator<Item = &'a GpuValue>,
) -> Vec<&'a GpuValue> {
    flattened_values(values)
        .into_iter()
        .filter(|value| matches!(value, GpuValue::Var(var) if runtime_type(var).is_some()))
        .collect()
}
