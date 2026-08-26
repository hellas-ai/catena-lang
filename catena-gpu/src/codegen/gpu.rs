use std::collections::BTreeSet;

use hexpr::Operation;
use thiserror::Error;

use super::{
    GpuAssign, GpuDialect, GpuFunction, GpuModuleMap, GpuValue, GpuVar, lower_types::CType, ops,
    runtime_type,
};

#[derive(Debug, Error)]
pub enum GpuRenderError {
    #[error("unsupported GPU operation `{0}`")]
    UnsupportedOp(Operation),
    #[error("operation `{op}` expected {expected} inputs and {outputs} outputs")]
    InvalidArity {
        op: Operation,
        expected: usize,
        outputs: usize,
    },
    #[error("invalid bool.ifc assignment: {0}")]
    InvalidIfc(&'static str),
    #[error("invalid gpu.launch assignment: {0}")]
    InvalidLaunch(&'static str),
    #[error("invalid fold assignment: {0}")]
    InvalidFold(&'static str),
}

pub fn render_modules(
    modules: &GpuModuleMap,
    dialect: GpuDialect,
) -> Result<String, GpuRenderError> {
    let mut output = render_prelude(dialect);
    for module in modules.values() {
        output.push_str(&signature(&module.entry, placement(&module.entry)));
        output.push_str(";\n");
    }
    output.push('\n');
    for module in modules.values() {
        for (assignment_index, assignment) in module.entry.assignments.iter().enumerate() {
            if assignment.op.as_str() == "gpu.launch" {
                ops::launch::render_kernel(
                    &mut output,
                    modules,
                    &module.entry,
                    assignment_index,
                    assignment,
                )?;
                output.push('\n');
            }
        }
    }
    for module in modules.values() {
        render_function(&mut output, &module.entry)?;
        output.push('\n');
    }
    Ok(output)
}

fn signature(function: &GpuFunction, placement: Placement) -> String {
    let mut params = function.sources.iter().map(param).collect::<Vec<_>>();
    params.extend(
        function
            .targets
            .iter()
            .enumerate()
            .map(|(index, var)| format!("{} *out_{index}", c_type(runtime_type(var).unwrap()))),
    );
    let qualifiers = match placement {
        Placement::HostOnly => "__host__",
        Placement::HostAndDevice => "__host__ __device__",
    };
    let generic_types = generic_types(function);
    let template = if generic_types.is_empty() {
        String::new()
    } else {
        format!(
            "template<{}>\n",
            generic_types
                .iter()
                .map(|index| format!("typename T{index}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let linkage = if generic_types.is_empty() {
        "extern \"C\" "
    } else {
        ""
    };
    format!(
        "{template}{linkage}{qualifiers} void {}({})",
        function.name,
        params.join(", ")
    )
}

fn generic_types(function: &GpuFunction) -> BTreeSet<usize> {
    let mut output = BTreeSet::new();
    for var in function.sources.iter().chain(&function.targets) {
        collect_generic_types(runtime_type(var).unwrap(), &mut output);
    }
    output
}

fn collect_generic_types(ty: &CType, output: &mut BTreeSet<usize>) {
    match ty {
        CType::Generic(index) => {
            output.insert(*index);
        }
        CType::Ptr(element) => collect_generic_types(element, output),
        _ => {}
    }
}

#[derive(Debug, Clone, Copy)]
enum Placement {
    HostOnly,
    HostAndDevice,
}

fn placement(function: &GpuFunction) -> Placement {
    if function
        .assignments
        .iter()
        .any(|assignment| assignment.op.as_str() == "gpu.launch")
    {
        Placement::HostOnly
    } else {
        Placement::HostAndDevice
    }
}

fn param(var: &GpuVar) -> String {
    format!("{} {}", c_type(runtime_type(var).unwrap()), var.name)
}

pub(super) fn c_type(ty: &CType) -> String {
    match ty {
        CType::Bool => "uint8_t".into(),
        CType::U32 => "uint32_t".into(),
        CType::U64 => "uint64_t".into(),
        CType::F32 => "float".into(),
        CType::Grid => "catena_grid_t".into(),
        CType::MemOwn => "catena_mem_own_t".into(),
        CType::Ptr(element) => format!("{} *", c_type(element)),
        CType::Generic(index) => format!("T{index}"),
        CType::Ix => "catena_ix_t".into(),
        CType::Thread => "catena_thread_t".into(),
        CType::Block => "catena_block_t".into(),
        CType::Scheduling => "catena_scheduling_t".into(),
    }
}

fn render_prelude(dialect: GpuDialect) -> String {
    let (error_type, success, error_string, synchronize) = match dialect {
        GpuDialect::Hip => (
            "hipError_t",
            "hipSuccess",
            "hipGetErrorString",
            "hipDeviceSynchronize",
        ),
        GpuDialect::Cuda => (
            "cudaError_t",
            "cudaSuccess",
            "cudaGetErrorString",
            "cudaDeviceSynchronize",
        ),
    };
    format!(
        r#"#include <{runtime_header}>
#include <stdint.h>
#include <math.h>
#include <stdio.h>

typedef struct {{ uint32_t x; uint32_t y; uint32_t z; }} catena_dim3_t;
typedef struct {{ catena_dim3_t grid_dim; catena_dim3_t block_dim; }} catena_grid_t;
typedef struct {{ void *data; uint64_t len; }} catena_mem_own_t;
typedef struct {{ uint64_t first; uint64_t second; uint64_t third; uint64_t linear; }} catena_ix_t;
typedef struct {{
    catena_ix_t global_index;
    catena_ix_t in_block_index;
    catena_ix_t block_index;
}} catena_thread_t;
typedef struct {{ catena_ix_t index; }} catena_block_t;
typedef enum {{
    CATENA_CELL_INACCESSIBLE = 0,
    CATENA_CELL_READABLE = 1,
    CATENA_CELL_OWNED = 2,
}} catena_cell_access_t;
typedef enum {{
    CATENA_SCHEDULING_OWN_EACH = 1,
    CATENA_SCHEDULING_READ_ALL = 2,
    CATENA_SCHEDULING_OWN_MATRIX_2D = 3,
}} catena_scheduling_kind_t;
typedef struct {{ catena_scheduling_kind_t kind; uint64_t matrix_columns; }} catena_scheduling_t;

__host__ __device__ static inline catena_cell_access_t catena_scheduling_resolve(
    catena_scheduling_t scheduling,
    catena_thread_t thread,
    catena_ix_t cell
) {{
    switch (scheduling.kind) {{
    case CATENA_SCHEDULING_OWN_EACH:
        return thread.global_index.linear == cell.linear
            ? CATENA_CELL_OWNED
            : CATENA_CELL_INACCESSIBLE;
    case CATENA_SCHEDULING_READ_ALL:
        return CATENA_CELL_READABLE;
    case CATENA_SCHEDULING_OWN_MATRIX_2D:
        if (scheduling.matrix_columns == 0) return CATENA_CELL_INACCESSIBLE;
        return thread.global_index.first == cell.linear % scheduling.matrix_columns
            && thread.global_index.second == cell.linear / scheduling.matrix_columns
            ? CATENA_CELL_OWNED
            : CATENA_CELL_INACCESSIBLE;
    default:
        return CATENA_CELL_INACCESSIBLE;
    }}
}}

__host__ static inline void catena_gpu_check({error_type} error) {{
    if (error != {success}) {{
        fprintf(stderr, "catena GPU error: %s\n", {error_string}(error));
        fflush(stderr);
        __builtin_trap();
    }}
}}

__host__ static inline void catena_gpu_synchronize(void) {{
    catena_gpu_check({synchronize}());
}}

"#,
        runtime_header = dialect.runtime_header(),
    )
}

fn render_function(output: &mut String, function: &GpuFunction) -> Result<(), GpuRenderError> {
    output.push_str(&signature(function, placement(function)));
    output.push_str(" {\n");
    let mut declared = function
        .sources
        .iter()
        .map(|var| var.name.clone())
        .collect::<BTreeSet<_>>();
    for (assignment_index, assignment) in function.assignments.iter().enumerate() {
        for var in &assignment.outputs {
            if declared.insert(var.name.clone()) {
                output.push_str(&format!(
                    "    {} {};\n",
                    c_type(runtime_type(var).unwrap()),
                    var.name
                ));
            }
        }
        render_assignment(output, function, assignment_index, assignment)?;
    }
    for (index, target) in function.targets.iter().enumerate() {
        output.push_str(&format!("    *out_{index} = {};\n", target.name));
    }
    output.push_str("}\n");
    Ok(())
}

fn render_assignment(
    output: &mut String,
    function: &GpuFunction,
    assignment_index: usize,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    if let Some(symbol) = &assignment.call_symbol {
        let mut args = assignment.inputs.iter().map(value_expr).collect::<Vec<_>>();
        args.extend(
            assignment
                .outputs
                .iter()
                .map(|var| format!("&{}", var.name)),
        );
        output.push_str(&format!("    {symbol}({});\n", args.join(", ")));
        return Ok(());
    }
    if ops::memory::render(output, assignment)?
        || ops::indexing::render(output, assignment)?
        || ops::scheduling::render(output, assignment)?
    {
        return Ok(());
    }
    let op = assignment.op.as_str();
    match op {
        "assert" => {
            if assignment.inputs.len() != 1 || !assignment.outputs.is_empty() {
                return Err(arity(assignment, 1));
            }
            output.push_str(&format!(
                "    if (!{}) __builtin_trap();\n",
                value_expr(&assignment.inputs[0])
            ));
            Ok(())
        }
        ":.param" => Ok(()),
        ":.ty"
        | ":.forget"
        | "bool.name"
        | "u64.name"
        | "gpu.thread.name"
        | "gpu.global.element-type" => identity(output, assignment),
        "u64.copies2" | "bool.copies2" | "ix.copies2" => copy_two(output, assignment),
        "gpu.grid.1d.intro" => render_grid_intro(output, assignment, 1),
        "gpu.grid.2d.intro" => render_grid_intro(output, assignment, 2),
        "gpu.grid.3d.intro" => render_grid_intro(output, assignment, 3),
        "gpu.launch" => ops::launch::render_call(output, function, assignment_index, assignment),
        "fold" => ops::fold::render(output, assignment),
        "bool.ifc" => render_ifc(output, assignment),
        "bool.t" => unary_output(output, assignment, "1", 0),
        "bool.f" | "u64.zero" | "u32.zero" | "scalar.zero" => {
            unary_output(output, assignment, "0", 0)
        }
        "u64.one" | "u32.one" | "f32.one" => unary_output(output, assignment, "1", 0),
        "bool.not" => unary(output, assignment, "!"),
        "u32.not" => unary(output, assignment, "~"),
        "f32.neg" => unary(output, assignment, "-"),
        "bool.and" => binary(output, assignment, "&&"),
        "bool.or" => binary(output, assignment, "||"),
        "u64.add" | "u32.add" | "f32.add" | "scalar.add" => binary(output, assignment, "+"),
        "u64.sub" | "u32.sub" | "f32.sub" => binary(output, assignment, "-"),
        "u64.mul" | "u32.mul" | "f32.mul" | "scalar.mul" => binary(output, assignment, "*"),
        "u64.div" | "f32.div" => binary(output, assignment, "/"),
        "u64.rem" => binary(output, assignment, "%"),
        "u32.and" => binary(output, assignment, "&"),
        "u32.or" => binary(output, assignment, "|"),
        "u32.xor" => binary(output, assignment, "^"),
        "u32.shl" => binary(output, assignment, "<<"),
        "u32.shr" => binary(output, assignment, ">>"),
        "u64.ne" | "u32.ne" | "f32.ne" => binary(output, assignment, "!="),
        "u32.eq" | "f32.eq" => binary(output, assignment, "=="),
        "u64.lt" | "u32.lt" | "f32.lt" => binary(output, assignment, "<"),
        "u32.gt" | "f32.gt" => binary(output, assignment, ">"),
        "u64.lte" | "u32.lte" | "f32.lte" => binary(output, assignment, "<="),
        "u64.gte" | "u32.gte" | "f32.gte" => binary(output, assignment, ">="),
        "bool.select" | "u64.select" | "u32.select" | "f32.select" => select(output, assignment),
        "u32.to-f32" => cast(output, assignment, "float"),
        "u64.to-f32" => cast(output, assignment, "float"),
        "u32.bitcast-f32" => bitcast(output, assignment, "float", "uint32_t"),
        "f32.bitcast-u32" => bitcast(output, assignment, "uint32_t", "float"),
        "f32.round-to-u32" => call1(output, assignment, "(uint32_t)roundf"),
        "f32.fma" => call3(output, assignment, "fmaf"),
        _ => Err(GpuRenderError::UnsupportedOp(assignment.op.clone())),
    }
}

fn render_grid_intro(
    output: &mut String,
    a: &GpuAssign,
    dimensions: usize,
) -> Result<(), GpuRenderError> {
    let expected = dimensions * 2;
    if a.inputs.len() != expected || a.outputs.len() != 2 {
        return Err(arity(a, expected));
    }
    let mut grid = ["1".to_string(), "1".to_string(), "1".to_string()];
    let mut block = ["1".to_string(), "1".to_string(), "1".to_string()];
    for axis in 0..dimensions {
        grid[axis] = input(a, axis)?;
        block[axis] = input(a, dimensions + axis)?;
    }
    output.push_str(&format!(
        "    {} = {{ {{ (uint32_t){}, (uint32_t){}, (uint32_t){} }}, {{ (uint32_t){}, (uint32_t){}, (uint32_t){} }} }};\n",
        a.outputs[0].name,
        grid[0], grid[1], grid[2], block[0], block[1], block[2],
    ));
    output.push_str(&format!(
        "    {} = ({} * {}) * ({} * {}) * ({} * {});\n",
        a.outputs[1].name, grid[0], block[0], grid[1], block[1], grid[2], block[2],
    ));
    Ok(())
}

fn identity(output: &mut String, a: &GpuAssign) -> Result<(), GpuRenderError> {
    unary_output(output, a, &input(a, 0)?, 1)
}

fn copy_two(output: &mut String, a: &GpuAssign) -> Result<(), GpuRenderError> {
    if a.inputs.len() != 1 || a.outputs.len() != 2 {
        return Err(arity(a, 1));
    }
    let value = input(a, 0)?;
    output.push_str(&format!(
        "    {} = {value};\n    {} = {value};\n",
        a.outputs[0].name, a.outputs[1].name,
    ));
    Ok(())
}

fn unary_output(
    output: &mut String,
    a: &GpuAssign,
    expression: &str,
    inputs: usize,
) -> Result<(), GpuRenderError> {
    if a.inputs.len() != inputs || a.outputs.len() != 1 {
        return Err(arity(a, inputs));
    }
    output.push_str(&format!("    {} = {expression};\n", a.outputs[0].name));
    Ok(())
}

fn unary(output: &mut String, a: &GpuAssign, operator: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("{operator}{}", input(a, 0)?), 1)
}

fn binary(output: &mut String, a: &GpuAssign, operator: &str) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!("{} {operator} {}", input(a, 0)?, input(a, 1)?),
        2,
    )
}

fn select(output: &mut String, a: &GpuAssign) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!("{} ? {} : {}", input(a, 0)?, input(a, 1)?, input(a, 2)?),
        3,
    )
}

fn cast(output: &mut String, a: &GpuAssign, ty: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("({ty}){}", input(a, 0)?), 1)
}

fn call1(output: &mut String, a: &GpuAssign, function: &str) -> Result<(), GpuRenderError> {
    unary_output(output, a, &format!("{function}({})", input(a, 0)?), 1)
}

fn call3(output: &mut String, a: &GpuAssign, function: &str) -> Result<(), GpuRenderError> {
    unary_output(
        output,
        a,
        &format!(
            "{function}({}, {}, {})",
            input(a, 0)?,
            input(a, 1)?,
            input(a, 2)?
        ),
        3,
    )
}

fn bitcast(output: &mut String, a: &GpuAssign, to: &str, from: &str) -> Result<(), GpuRenderError> {
    if a.inputs.len() != 1 || a.outputs.len() != 1 {
        return Err(arity(a, 1));
    }
    output.push_str(&format!(
        "    {{ union {{ {from} from; {to} to; }} bits = {{ {} }}; {} = bits.to; }}\n",
        value_expr(&a.inputs[0]),
        a.outputs[0].name
    ));
    Ok(())
}

fn input(a: &GpuAssign, index: usize) -> Result<String, GpuRenderError> {
    a.inputs
        .get(index)
        .map(value_expr)
        .ok_or_else(|| arity(a, index + 1))
}

pub(super) fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(target) => sanitize_ident(&format!("program.{target}")),
    }
}

fn render_ifc(output: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let function_positions = assignment
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(index, value)| matches!(value, GpuValue::FnSymbol(_)).then_some(index))
        .collect::<Vec<_>>();
    let [true_fn, false_fn] = function_positions.as_slice() else {
        return Err(GpuRenderError::InvalidIfc(
            "expected exactly two direct function names",
        ));
    };
    let tail = assignment
        .inputs
        .get(false_fn + 1..)
        .ok_or(GpuRenderError::InvalidIfc("missing decision Boolean"))?;
    let [flag, arguments @ ..] = tail else {
        return Err(GpuRenderError::InvalidIfc("missing decision Boolean"));
    };

    output.push_str(&format!("    if ({}) {{\n", value_expr(flag)));
    render_ifc_call(
        output,
        &assignment.inputs[*true_fn],
        assignment.inputs[..*true_fn].iter().chain(arguments.iter()),
        &assignment.outputs,
    );
    output.push_str("    } else {\n");
    render_ifc_call(
        output,
        &assignment.inputs[*false_fn],
        assignment.inputs[true_fn + 1..*false_fn]
            .iter()
            .chain(arguments.iter()),
        &assignment.outputs,
    );
    output.push_str("    }\n");
    Ok(())
}

fn render_ifc_call<'a>(
    output: &mut String,
    function: &GpuValue,
    inputs: impl Iterator<Item = &'a GpuValue>,
    outputs: &[GpuVar],
) {
    let mut arguments = inputs.map(value_expr).collect::<Vec<_>>();
    arguments.extend(outputs.iter().map(|var| format!("&{}", var.name)));
    output.push_str(&format!(
        "        {}({});\n",
        value_expr(function),
        arguments.join(", ")
    ));
}

pub(super) fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn arity(a: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidArity {
        op: a.op.clone(),
        expected,
        outputs: a.outputs.len(),
    }
}

pub(super) fn invalid_arity(
    assignment: &GpuAssign,
    expected_inputs: usize,
    _expected_outputs: usize,
) -> GpuRenderError {
    arity(assignment, expected_inputs)
}

#[cfg(test)]
mod tests {
    use open_hypergraphs::lax::NodeId;

    use super::*;
    use crate::codegen::lower_types::LoweredType;

    #[test]
    fn lower_rank_grids_are_padded_only_in_generated_code() {
        let one_d = grid_assignment("gpu.grid.1d.intro", &["gx", "bx"]);
        let mut generated = String::new();
        render_grid_intro(&mut generated, &one_d, 1).unwrap();
        assert!(generated.contains("{ (uint32_t)gx, (uint32_t)1, (uint32_t)1 }"));
        assert!(generated.contains("{ (uint32_t)bx, (uint32_t)1, (uint32_t)1 }"));

        let two_d = grid_assignment("gpu.grid.2d.intro", &["gx", "gy", "bx", "by"]);
        generated.clear();
        render_grid_intro(&mut generated, &two_d, 2).unwrap();
        assert!(generated.contains("{ (uint32_t)gx, (uint32_t)gy, (uint32_t)1 }"));
        assert!(generated.contains("{ (uint32_t)bx, (uint32_t)by, (uint32_t)1 }"));
    }

    fn grid_assignment(operation: &str, inputs: &[&str]) -> GpuAssign {
        GpuAssign {
            op: operation.parse().unwrap(),
            call_symbol: None,
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(node, name)| GpuValue::Var(var(node, name, CType::U64)))
                .collect(),
            outputs: vec![
                var(inputs.len(), "grid", CType::Grid),
                var(inputs.len() + 1, "size", CType::U64),
            ],
        }
    }

    fn var(node: usize, name: &str, ty: CType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.into(),
            lowered: LoweredType::Runtime(ty),
        }
    }
}
