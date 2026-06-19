use crate::codegen::{GpuAssign, GpuVar, gpu::GpuRenderError, lower_types::CType, runtime_type};

pub(in crate::codegen) fn param_decl(
    var: &GpuVar,
    by_pointer: bool,
) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    if by_pointer {
        Ok(format!("{} *out_{}", c_type(ty), var.name))
    } else {
        Ok(format!("{} {}", c_type(ty), var.name))
    }
}

pub(in crate::codegen) fn c_type(ty: &CType) -> String {
    match ty {
        CType::Unit => "catena_unit_t".to_string(),
        CType::Bool => "uint8_t".to_string(),
        CType::U32 => "uint32_t".to_string(),
        CType::U64 => "uint64_t".to_string(),
        CType::F32 => "float".to_string(),
        CType::Pointer(inner) => format!("{} *", c_type(inner)),
        CType::Named(name) => name.clone(),
    }
}

pub(in crate::codegen) fn invalid_inputs(
    assignment: &GpuAssign,
    expected: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidInputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.inputs.len(),
    }
}

pub(in crate::codegen) fn invalid_outputs(
    assignment: &GpuAssign,
    expected: usize,
) -> GpuRenderError {
    GpuRenderError::InvalidOutputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.outputs.len(),
    }
}

pub(in crate::codegen) fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}
