//! Backend-independent decoding of generic `parallel.*` operations.
//!
//! This module understands the flattened ABI of parallel operations, but it
//! does not decide how a topology allocates, synchronizes, or executes work.
//! Those decisions belong to backend modules such as [`gpu`].

pub(in crate::codegen) mod gpu;

use crate::codegen::{
    GpuAssign, GpuValue, GpuVar,
    components::{input_components, single_function, single_value},
    gpu::GpuRenderError,
    runtime_type,
};

pub(super) struct MaterializeParts<'a> {
    pub(super) runner: &'a GpuValue,
    pub(super) buffer: &'a GpuValue,
    pub(super) buffer_var: &'a GpuVar,
    pub(super) environment: &'a [GpuValue],
    pub(super) function: &'a GpuValue,
}

/// Decode the generic `parallel.materializec` ABI:
///
/// ```text
/// runner, writable buffer, environment, producer
///     -> runner, pending buffer
/// ```
pub(super) fn materialize_parts(
    assignment: &GpuAssign,
) -> Result<MaterializeParts<'_>, GpuRenderError> {
    let components = input_components(assignment)?;
    let [runner, buffer, environment, function] = components.as_slice() else {
        return Err(GpuRenderError::InvalidInputComponentCount {
            op: assignment.op.clone(),
            expected: 4,
            actual: components.len(),
        });
    };
    let runner = single_value(runner)
        .map_err(|error| component_error(assignment, "runner", error.actual))?;
    let buffer = single_value(buffer)
        .map_err(|error| component_error(assignment, "buffer", error.actual))?;
    let GpuValue::Var(buffer_var) = buffer else {
        unreachable!("single runtime value must be a variable")
    };
    let function = single_function(function)
        .map_err(|error| component_error(assignment, "producer", error.actual))?;
    Ok(MaterializeParts {
        runner,
        buffer,
        buffer_var,
        environment,
        function,
    })
}

pub(super) struct PreservedRuntimeValues<'a> {
    pub(super) inputs: Vec<&'a GpuValue>,
    pub(super) outputs: Vec<&'a GpuVar>,
}

/// Extract the runtime state threaded through generic synchronization ops.
/// Type-level topology and dependency witnesses erase before this point.
pub(super) fn preserved_runtime_values(
    assignment: &GpuAssign,
) -> Result<PreservedRuntimeValues<'_>, GpuRenderError> {
    let inputs = assignment
        .inputs
        .iter()
        .filter(|input| matches!(input, GpuValue::Var(var) if runtime_type(var).is_some()))
        .collect::<Vec<_>>();
    let outputs = assignment
        .outputs
        .iter()
        .filter(|output| runtime_type(output).is_some())
        .collect::<Vec<_>>();
    if inputs.len() != outputs.len() {
        return Err(crate::codegen::render_utils::invalid_outputs(
            assignment,
            inputs.len(),
        ));
    }
    Ok(PreservedRuntimeValues { inputs, outputs })
}

fn component_error(
    assignment: &GpuAssign,
    component: &'static str,
    actual: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputComponentValueCount {
        op: assignment.op.clone(),
        component,
        description: "runtime value",
        expected: 1,
        actual,
    }
}
