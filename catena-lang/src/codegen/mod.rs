//! GPU codegen.
//!
//! This module lowers closure-converted, typed hypergraphs into a small dataflow
//! GPU artifact. Report generation should render this artifact, not make codegen
//! decisions itself.

pub mod fn_ptrs;
pub mod gpu;
pub mod lower_types;
mod ops;
mod prelude;
mod render_utils;
mod specialize;
mod validate;

use std::{
    collections::{BTreeMap, VecDeque},
    ops::Range,
};

use hexpr::Operation;
use metacat::{
    check::{Error as MetacatCheckError, eval_type},
    dual,
    ssa::{SSAError, ssa},
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    codegen::{
        fn_ptrs::{FnPtrSymbol, FnPtrSymbolError, direct_fn_ptr_symbols},
        lower_types::{CType, LowerTypeError, LoweredType, lower_type},
        specialize::{
            PendingInstance, SpecializationKey, entrypoint_key, specialization_key,
            specialization_overrides,
        },
    },
    report::{AnnotatedTerm, TheoryTermMap},
};

pub type GpuModuleMap = BTreeMap<Operation, GpuModule>;

const PROGRAM_THEORY: &str = "program";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDialect {
    Hip,
    Cuda,
}

impl GpuDialect {
    pub fn runtime_header(self) -> &'static str {
        match self {
            Self::Hip => "hip/hip_runtime.h",
            Self::Cuda => "cuda_runtime.h",
        }
    }

    pub fn error_type(self) -> &'static str {
        match self {
            Self::Hip => "hipError_t",
            Self::Cuda => "cudaError_t",
        }
    }

    pub fn success_value(self) -> &'static str {
        match self {
            Self::Hip => "hipSuccess",
            Self::Cuda => "cudaSuccess",
        }
    }

    pub fn error_string_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipGetErrorString",
            Self::Cuda => "cudaGetErrorString",
        }
    }

    pub fn managed_alloc_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipMallocManaged",
            Self::Cuda => "cudaMallocManaged",
        }
    }

    pub fn synchronize_fn(self) -> &'static str {
        match self {
            Self::Hip => "hipDeviceSynchronize",
            Self::Cuda => "cudaDeviceSynchronize",
        }
    }

    pub fn device_compile_guard(self) -> &'static str {
        match self {
            Self::Hip => "__HIP_DEVICE_COMPILE__",
            Self::Cuda => "__CUDA_ARCH__",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuModule {
    /// generated code symbol
    pub name: String,
    /// Corresponding source name (if applicable)
    pub source_name: Option<Operation>,
    /// Definition
    pub entry: GpuFunction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuFunction {
    pub name: String,
    pub sources: Vec<GpuVar>,
    pub targets: Vec<GpuVar>,
    pub assignments: Vec<GpuAssign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAssign {
    pub op: Operation,
    pub call_symbol: Option<String>,
    /// Flattened codegen inputs passed to renderers.
    pub inputs: Vec<GpuValue>,
    /// One group per original Catena/SSA source object before input flattening.
    pub source_input_groups: Vec<InputGroup>,
    pub outputs: Vec<GpuVar>,
}

impl GpuAssign {
    pub fn source_groups(&self) -> &[InputGroup] {
        &self.source_input_groups
    }

    pub fn group_values(&self, group: &InputGroup) -> &[GpuValue] {
        &self.inputs[group.inputs.clone()]
    }

    pub fn single_group_value(&self, group: &InputGroup) -> Option<&GpuValue> {
        let [value] = self.group_values(group) else {
            return None;
        };
        Some(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputGroup {
    /// Position in the declared source object list before flattening/expansion.
    pub source_index: usize,
    /// Original graph node for this source object when it expands to exactly one flat input.
    pub node: Option<NodeId>,
    /// Original source type before runtime/codegen flattening/expansion.
    pub ty: Tree<(), Operation>,
    /// Slice in `GpuAssign::inputs` produced by this source object.
    pub inputs: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuValue {
    Var(GpuVar),
    FnSymbol(FnPtrSymbol),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuVar {
    pub node: NodeId,
    pub name: String,
    pub lowered: LoweredType,
}

struct CodegenState<'a> {
    definitions: &'a BTreeMap<Operation, AnnotatedTerm>,
    source_object_types: BTreeMap<Operation, Vec<Tree<(), Operation>>>,
    modules: GpuModuleMap,
    instances: BTreeMap<(Operation, SpecializationKey), String>,
    queue: VecDeque<PendingInstance>,
    next_specialization_id: usize,
}

/// Codegen for all functions, producing per-definition GPU modules.
pub fn codegen(
    terms: &TheoryTermMap,
    theory_set: &TheorySet,
) -> Result<GpuModuleMap, CodegenError> {
    let theory_id = TheoryId(
        PROGRAM_THEORY
            .parse()
            .expect("program theory id should parse"),
    );
    let Some(definitions) = terms.get(&theory_id) else {
        return Ok(BTreeMap::new());
    };

    let mut state = CodegenState {
        definitions,
        source_object_types: program_source_object_types(theory_set)?,
        modules: BTreeMap::new(),
        instances: BTreeMap::new(),
        queue: VecDeque::new(),
        next_specialization_id: 0,
    };

    for (definition_name, term) in definitions {
        let Some(key) = entrypoint_key(term)? else {
            continue;
        };
        let source_name = definition_name.clone();
        let name = sanitize_ident(&format!("{theory_id}.{definition_name}"));
        state
            .instances
            .insert((definition_name.clone(), key.clone()), name.clone());
        state.queue.push_back(PendingInstance {
            op: definition_name.clone(),
            name,
            source_name: Some(source_name),
            overrides: BTreeMap::new(),
        });
    }

    while let Some(instance) = state.queue.pop_front() {
        let module_key: Operation = instance
            .name
            .parse()
            .expect("generated function name should parse as operation");
        if state.modules.contains_key(&module_key) {
            continue;
        }
        let term = state
            .definitions
            .get(&instance.op)
            .expect("queued specialization should have a definition");
        let module = state.codegen_definition(term, &instance)?;
        state.modules.insert(module_key, module);
    }

    Ok(state.modules)
}

fn program_source_object_types(
    theory_set: &TheorySet,
) -> Result<BTreeMap<Operation, Vec<Tree<(), Operation>>>, CodegenError> {
    let theory_id = TheoryId(
        PROGRAM_THEORY
            .parse()
            .expect("program theory id should parse"),
    );
    let Some(Theory::Theory { arrows, .. }) = theory_set.theories.get(&theory_id) else {
        return Err(CodegenError::MissingProgramTheory);
    };

    arrows
        .iter()
        .map(|(name, arrow)| {
            let source_map = arrow.type_maps.0.clone();
            let mut type_term = dual::into_fwd(source_map.clone());
            let q = type_term
                .quotient()
                .map_err(|error| CodegenError::SourceTypeMap {
                    operation: name.clone(),
                    error: MetacatCheckError::InvalidQuotient(error),
                })?;
            let types = eval_type(type_term).map_err(|error| CodegenError::SourceTypeMap {
                operation: name.clone(),
                error,
            })?;
            let source_types = source_map
                .targets
                .iter()
                .map(|target| types[q.table[target.0]].clone())
                .collect::<Vec<_>>();
            Ok((name.clone(), source_types))
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ssa(#[from] SSAError),
    #[error("failed to quotient transformed term before codegen: {0:?}")]
    Quotient(open_hypergraphs::strict::vec::FiniteFunction),
    #[error(transparent)]
    LowerType(#[from] LowerTypeError),
    #[error(transparent)]
    FnPtrSymbol(#[from] FnPtrSymbolError),
    #[error("missing program theory during GPU codegen")]
    MissingProgramTheory,
    #[error("failed to evaluate source type map for `{operation}`: {error:?}")]
    SourceTypeMap {
        operation: Operation,
        error: MetacatCheckError<Operation>,
    },
    #[error("missing declared source object types for operation `{0}` during GPU codegen")]
    MissingSourceObjectTypes(Operation),
    #[error(
        "failed to align source groups for `{operation}`: {declared} declared source objects, {actual} flat inputs, consumed {consumed}"
    )]
    SourceGroupAlignment {
        operation: Operation,
        declared: usize,
        actual: usize,
        consumed: usize,
    },
    #[error("definition `{0}` is used with non-monomorphic runtime interface")]
    NonMonomorphicUse(Operation),
    #[error(
        "definition `{caller}` uses `{producer}` as a materializec producer, but device-callable producer dependency `{containing}` contains `{nested}`. materializec lowering is host-only: it allocates output memory and launches a GPU kernel. A materializec producer is called from GPU device code, so it and the program definitions it calls must be device-callable and allocation-free. Move the nested materialization out of the producer call chain, or pass a precomputed buffer as the producer environment."
    )]
    MaterializecProducerContainsMaterialize {
        caller: Operation,
        producer: Operation,
        containing: Operation,
        nested: Operation,
    },
}

impl CodegenState<'_> {
    /// Lower one closure-converted, type-annotated definition into the dataflow GPU artifact.
    ///
    /// Direct `name.*` producers are recorded as symbolic function values instead of runtime
    /// assignments. Calls to other `program` definitions are resolved to generated specialization
    /// symbols and enqueue those specializations as needed.
    fn codegen_definition(
        &mut self,
        term: &AnnotatedTerm,
        instance: &PendingInstance,
    ) -> Result<GpuModule, CodegenError> {
        let fn_symbols = direct_fn_ptr_symbols(term)?;

        let mut term = term.clone();
        term.quotient().map_err(CodegenError::Quotient)?;

        let mut sources = Vec::new();
        for source in &term.sources {
            let var = var(*source, &term, &instance.overrides)?;
            if matches!(var.lowered, LoweredType::Runtime(_)) {
                sources.push(var);
            }
        }

        let mut targets = Vec::new();
        for target in &term.targets {
            let var = var(*target, &term, &instance.overrides)?;
            if matches!(var.lowered, LoweredType::Runtime(_)) {
                targets.push(var);
            }
        }

        let mut assignments = Vec::new();
        for assignment in ssa(term.clone().to_strict())? {
            if assignment.op.as_str().starts_with("name.") {
                continue;
            }

            let actual_source_types = assignment
                .sources
                .iter()
                .map(|(_, ty)| ty.clone())
                .collect::<Vec<_>>();
            let declared_source_types = self
                .source_object_types
                .get(&assignment.op)
                .ok_or_else(|| CodegenError::MissingSourceObjectTypes(assignment.op.clone()))?;
            let source_ranges =
                source_group_ranges(&assignment.op, &declared_source_types, &actual_source_types)?;
            let mut source_input_groups = Vec::new();
            let mut inputs = Vec::new();
            for (source_index, (ty, source_range)) in declared_source_types
                .iter()
                .zip(source_ranges.iter())
                .enumerate()
            {
                let start = inputs.len();
                for (node, _) in &assignment.sources[source_range.clone()] {
                    let input = if let Some(symbol) = fn_symbols.get(node) {
                        GpuValue::FnSymbol(symbol.clone())
                    } else {
                        GpuValue::Var(var(*node, &term, &instance.overrides)?)
                    };
                    inputs.push(input);
                }
                let node = if source_range.len() == 1 {
                    Some(assignment.sources[source_range.start].0)
                } else {
                    None
                };
                source_input_groups.push(InputGroup {
                    source_index,
                    node,
                    ty: ty.clone(),
                    inputs: start..inputs.len(),
                });
            }
            let outputs = assignment
                .targets
                .iter()
                .map(|(node, _)| var(*node, &term, &instance.overrides))
                .collect::<Result<Vec<_>, CodegenError>>()?;

            validate::assignment(&self.definitions, &instance.op, &assignment.op, &inputs)?;

            let call_symbol = if self.definitions.contains_key(&assignment.op) {
                Some(self.ensure_specialization(&assignment.op, &inputs, &outputs)?)
            } else {
                None
            };

            assignments.push(GpuAssign {
                op: assignment.op,
                call_symbol,
                inputs,
                source_input_groups,
                outputs,
            });
        }

        Ok(GpuModule {
            name: instance.name.clone(),
            source_name: instance.source_name.clone(),
            entry: GpuFunction {
                name: instance.name.clone(),
                sources,
                targets,
                assignments,
            },
        })
    }

    fn ensure_specialization(
        &mut self,
        op: &Operation,
        inputs: &[GpuValue],
        outputs: &[GpuVar],
    ) -> Result<String, CodegenError> {
        let key = specialization_key(inputs, outputs)
            .ok_or_else(|| CodegenError::NonMonomorphicUse(op.clone()))?;
        if let Some(name) = self.instances.get(&(op.clone(), key.clone())) {
            return Ok(name.clone());
        }

        let name = sanitize_ident(&format!(
            "{PROGRAM_THEORY}.{op}__{}",
            self.next_specialization_id
        ));
        self.next_specialization_id += 1;
        let overrides = specialization_overrides(
            self.definitions
                .get(op)
                .expect("specialized operation should have a definition"),
            inputs,
            outputs,
        );
        self.instances
            .insert((op.clone(), key.clone()), name.clone());
        self.queue.push_back(PendingInstance {
            op: op.clone(),
            name: name.clone(),
            source_name: None,
            overrides,
        });
        Ok(name)
    }
}

type TypeSubstitution = BTreeMap<usize, Tree<(), Operation>>;

fn source_group_ranges(
    operation: &Operation,
    declared: &[Tree<(), Operation>],
    actual: &[Tree<(), Operation>],
) -> Result<Vec<Range<usize>>, CodegenError> {
    let Some((ranges, consumed)) =
        align_source_groups(declared, actual, 0, TypeSubstitution::new())
    else {
        return Err(CodegenError::SourceGroupAlignment {
            operation: operation.clone(),
            declared: declared.len(),
            actual: actual.len(),
            consumed: 0,
        });
    };
    if consumed != actual.len() {
        return Err(CodegenError::SourceGroupAlignment {
            operation: operation.clone(),
            declared: declared.len(),
            actual: actual.len(),
            consumed,
        });
    }
    Ok(ranges)
}

fn align_source_groups(
    declared: &[Tree<(), Operation>],
    actual: &[Tree<(), Operation>],
    offset: usize,
    substitution: TypeSubstitution,
) -> Option<(Vec<Range<usize>>, usize)> {
    let Some((source, rest)) = declared.split_first() else {
        return Some((Vec::new(), offset));
    };

    let mut matches = match_source_object(source, actual, substitution);
    matches.sort_by_key(|(consumed, _)| std::cmp::Reverse(*consumed));

    for (consumed, substitution) in matches {
        let start = offset;
        let end = offset + consumed;
        if let Some((mut ranges, final_offset)) =
            align_source_groups(rest, &actual[consumed..], end, substitution)
        {
            ranges.insert(0, start..end);
            return Some((ranges, final_offset));
        }
    }

    None
}

fn match_source_object(
    pattern: &Tree<(), Operation>,
    actual: &[Tree<(), Operation>],
    substitution: TypeSubstitution,
) -> Vec<(usize, TypeSubstitution)> {
    match pattern {
        Tree::Empty => vec![(0, substitution)],
        Tree::Leaf(index, _) => match substitution.get(index) {
            Some(bound) if is_unit_type(bound) => vec![(0, substitution)],
            Some(bound) => {
                if actual.first().is_some_and(|candidate| bound == candidate) {
                    vec![(1, substitution)]
                } else {
                    Vec::new()
                }
            }
            None => {
                let mut out = Vec::new();
                if let Some(candidate) = actual.first() {
                    let mut consume_substitution = substitution.clone();
                    consume_substitution.insert(*index, candidate.clone());
                    out.push((1, consume_substitution));
                }
                let mut unit_substitution = substitution;
                unit_substitution.insert(*index, unit_type());
                out.push((0, unit_substitution));
                out
            }
        },
        Tree::Node(op, _, children) if op.as_str() == "1" && children.is_empty() => {
            vec![(0, substitution)]
        }
        Tree::Node(op, _, children) if matches!(op.as_str(), "*" | "=>") => {
            match_source_object_sequence(children, actual, 0, substitution)
        }
        _ => actual
            .first()
            .and_then(|candidate| {
                let mut substitution = substitution;
                unify_type(pattern, candidate, &mut substitution).then_some((1, substitution))
            })
            .into_iter()
            .collect(),
    }
}

fn match_source_object_sequence(
    patterns: &[Tree<(), Operation>],
    actual: &[Tree<(), Operation>],
    consumed: usize,
    substitution: TypeSubstitution,
) -> Vec<(usize, TypeSubstitution)> {
    let Some((pattern, rest)) = patterns.split_first() else {
        return vec![(consumed, substitution)];
    };

    let mut out = Vec::new();
    for (next_consumed, substitution) in match_source_object(pattern, actual, substitution) {
        out.extend(match_source_object_sequence(
            rest,
            &actual[next_consumed..],
            consumed + next_consumed,
            substitution,
        ));
    }
    out
}

fn unify_type(
    pattern: &Tree<(), Operation>,
    actual: &Tree<(), Operation>,
    substitution: &mut TypeSubstitution,
) -> bool {
    match pattern {
        Tree::Empty => matches!(actual, Tree::Empty),
        Tree::Leaf(index, _) => match substitution.get(index) {
            Some(bound) => bound == actual,
            None => {
                substitution.insert(*index, actual.clone());
                true
            }
        },
        Tree::Node(pattern_op, pattern_port, pattern_children) => {
            let Tree::Node(actual_op, actual_port, actual_children) = actual else {
                return false;
            };
            pattern_op == actual_op
                && pattern_port == actual_port
                && pattern_children.len() == actual_children.len()
                && pattern_children.iter().zip(actual_children).all(
                    |(pattern_child, actual_child)| {
                        unify_type(pattern_child, actual_child, substitution)
                    },
                )
        }
    }
}

fn unit_type() -> Tree<(), Operation> {
    Tree::Node(
        "1".parse().expect("unit type operation should parse"),
        0,
        Vec::new(),
    )
}

fn is_unit_type(ty: &Tree<(), Operation>) -> bool {
    matches!(ty, Tree::Node(op, _, children) if op.as_str() == "1" && children.is_empty())
}

fn var(
    node: NodeId,
    term: &AnnotatedTerm,
    overrides: &BTreeMap<usize, LoweredType>,
) -> Result<GpuVar, CodegenError> {
    Ok(GpuVar {
        node,
        name: node_var(node),
        lowered: overrides
            .get(&node.0)
            .cloned()
            .unwrap_or(lower_type(&term.hypergraph.nodes[node.0])?),
    })
}

fn node_var(node: NodeId) -> String {
    format!("x{}", node.0)
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

pub fn runtime_type(var: &GpuVar) -> Option<&CType> {
    match &var.lowered {
        LoweredType::Runtime(ty) => Some(ty),
        LoweredType::Erased => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn leaf(index: usize) -> Tree<(), Operation> {
        Tree::Leaf(index, ())
    }

    fn node(name: &str, children: Vec<Tree<(), Operation>>) -> Tree<(), Operation> {
        Tree::Node(op(name), 0, children)
    }

    #[test]
    fn source_groups_preserve_unit_source_boundary() {
        let a = leaf(0);
        let x = leaf(1);
        let y = leaf(2);
        let declared = vec![
            a.clone(),
            x.clone(),
            node("fn", vec![x.clone(), a.clone()]),
            y.clone(),
        ];
        let actual = vec![
            node("u64", vec![]),
            node("fn", vec![unit_type(), node("u64", vec![])]),
            node("buf", vec![]),
        ];

        let ranges = source_group_ranges(&op("reducec"), &declared, &actual).unwrap();

        assert_eq!(ranges, vec![0..1, 1..1, 1..2, 2..3]);
    }

    #[test]
    fn source_groups_preserve_product_source_boundary() {
        let declared = vec![node("*", vec![leaf(0), leaf(1)]), leaf(2)];
        let actual = vec![
            node("u64", vec![]),
            node("bool", vec![]),
            node("buf", vec![]),
        ];

        let ranges = source_group_ranges(&op("pair_source"), &declared, &actual).unwrap();

        assert_eq!(ranges, vec![0..2, 2..3]);
    }
}
