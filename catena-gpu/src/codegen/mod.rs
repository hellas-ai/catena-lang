//! Scalar-only GPU code generation.

pub mod gpu;
pub mod lower_types;
mod ops;
mod specialize;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use hexpr::Operation;
use metacat::{
    ssa::{SSAError, ssa},
    theory::TheoryId,
    tree::Tree,
};
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

pub use catena_lang::codegen::GpuDialect;

use crate::{check::AnnotatedTerm, report::TheoryTermMap};
use lower_types::{CType, LowerTypeError, LoweredType, infer_type, lower_type};
use specialize::{PendingInstance, SpecializationKey, boundary_key};

pub type GpuModuleMap = BTreeMap<Operation, GpuModule>;
type CodegenTerm = AnnotatedTerm<Operation>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuModule {
    pub name: String,
    pub source_name: Option<Operation>,
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
    pub inputs: Vec<GpuValue>,
    pub outputs: Vec<GpuVar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuValue {
    Var(GpuVar),
    FnSymbol(Operation),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuVar {
    pub node: NodeId,
    pub name: String,
    pub lowered: LoweredType,
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ssa(#[from] SSAError),
    #[error("failed to quotient term before codegen: {0:?}")]
    Quotient(open_hypergraphs::strict::vec::FiniteFunction),
    #[error(transparent)]
    LowerType(#[from] LowerTypeError),
    #[error("generated function name `{operation}` expected one output, found {actual}")]
    InvalidNameArity { operation: Operation, actual: usize },
    #[error("generated function name `{0}` has an invalid target")]
    InvalidNameTarget(String),
    #[error("structural product operation `{operation}` has incompatible runtime components")]
    InvalidProduct { operation: Operation },
    #[error("a function symbol reached the runtime output of `{0}`")]
    FunctionOutput(Operation),
    #[error("kernel `{kernel}` is not device-callable: {path}")]
    NestedLaunch { kernel: Operation, path: String },
    #[error("definition `{0}` is used without a concrete runtime specialization")]
    NonMonomorphicUse(Operation),
    #[error(
        "definition `{operation}` has {formal} represented boundary values but was called with {actual}"
    )]
    SpecializationArity {
        operation: Operation,
        formal: usize,
        actual: usize,
    },
    #[error("named definition `{0}` does not have a function type")]
    InvalidFunctionType(Operation),
}

pub fn codegen(terms: &TheoryTermMap) -> Result<GpuModuleMap, CodegenError> {
    let program = TheoryId("program".parse().unwrap());
    let Some(definitions) = terms.get(&program) else {
        return Ok(BTreeMap::new());
    };
    let mut state = CodegenState {
        definitions,
        modules: BTreeMap::new(),
        instances: BTreeMap::new(),
        queue: VecDeque::new(),
        next_specialization_id: 0,
    };

    for (operation, term) in definitions {
        let Some((sources, targets)) = concrete_boundary(term)? else {
            continue;
        };
        let key = boundary_key(sources, targets);
        let symbol = sanitize_ident(&format!("program.{operation}"));
        state
            .instances
            .insert((operation.clone(), key), symbol.clone());
        state.queue.push_back(PendingInstance {
            operation: operation.clone(),
            symbol,
            source_name: Some(operation.clone()),
            substitutions: BTreeMap::new(),
        });
    }

    while let Some(instance) = state.queue.pop_front() {
        let module_key = instance
            .symbol
            .parse()
            .expect("generated specialization symbol should be an operation");
        if state.modules.contains_key(&module_key) {
            continue;
        }
        let term = state
            .definitions
            .get(&instance.operation)
            .expect("queued specialization should have a definition");
        let module = state.codegen_definition(term, &instance)?;
        state.modules.insert(module_key, module);
    }
    validate_device_callability(&state.modules)?;
    Ok(state.modules)
}

struct CodegenState<'a> {
    definitions: &'a BTreeMap<Operation, CodegenTerm>,
    modules: GpuModuleMap,
    instances: BTreeMap<(Operation, SpecializationKey), String>,
    queue: VecDeque<PendingInstance>,
    next_specialization_id: usize,
}

fn validate_device_callability(modules: &GpuModuleMap) -> Result<(), CodegenError> {
    for module in modules.values() {
        for assignment in &module.entry.assignments {
            if assignment.op.as_str() != "gpu.launch" {
                continue;
            }
            let Some(GpuValue::FnSymbol(kernel)) = assignment.inputs.last() else {
                continue;
            };
            let mut active = HashSet::new();
            let mut checked = HashSet::new();
            let mut path = Vec::new();
            validate_device_function(
                modules,
                kernel,
                kernel,
                &mut active,
                &mut checked,
                &mut path,
            )?;
        }
    }
    Ok(())
}

fn validate_device_function(
    modules: &GpuModuleMap,
    kernel: &Operation,
    function: &Operation,
    active: &mut HashSet<Operation>,
    checked: &mut HashSet<Operation>,
    path: &mut Vec<Operation>,
) -> Result<(), CodegenError> {
    if checked.contains(function) || !active.insert(function.clone()) {
        return Ok(());
    }
    let Some(module) = modules.get(function) else {
        active.remove(function);
        return Ok(());
    };
    path.push(function.clone());

    for assignment in &module.entry.assignments {
        if assignment.op.as_str() == "gpu.launch" {
            let mut call_path = path.iter().map(Operation::as_str).collect::<Vec<_>>();
            call_path.push("gpu.launch");
            return Err(CodegenError::NestedLaunch {
                kernel: kernel.clone(),
                path: call_path.join(" -> "),
            });
        }

        let mut callees = Vec::new();
        if assignment.call_symbol.is_some() {
            callees.push(&assignment.op);
        }
        callees.extend(assignment.inputs.iter().filter_map(|input| match input {
            GpuValue::FnSymbol(callee) => Some(callee),
            GpuValue::Var(_) => None,
        }));
        for callee in callees {
            validate_device_function(modules, kernel, callee, active, checked, path)?;
        }
    }

    path.pop();
    active.remove(function);
    checked.insert(function.clone());
    Ok(())
}

impl CodegenState<'_> {
    fn codegen_definition(
        &mut self,
        source_term: &CodegenTerm,
        instance: &PendingInstance,
    ) -> Result<GpuModule, CodegenError> {
        let mut term = source_term.clone();
        term.quotient().map_err(CodegenError::Quotient)?;
        let function_symbols = direct_function_symbols(&term)?;
        let mut aliases = HashMap::<NodeId, Vec<GpuValue>>::new();
        let mut sources = Vec::new();
        for node in &term.sources {
            let components = vars(*node, &term, &instance.substitutions)?;
            sources.extend(components.iter().cloned());
            aliases.insert(*node, components.into_iter().map(GpuValue::Var).collect());
        }
        let mut assignments = Vec::new();
        for assignment in ssa(term.clone().to_strict())? {
            let op = assignment.op;
            if op.as_str().starts_with("name.") {
                continue;
            }
            if op.as_str() == "*.intro" {
                let inputs = self.resolve_nodes(
                    assignment.sources.iter().map(|(node, _)| *node),
                    &aliases,
                    &function_symbols,
                    &term,
                    &instance.substitutions,
                )?;
                let [(target, _)] = assignment.targets.as_slice() else {
                    return Err(CodegenError::InvalidProduct { operation: op });
                };
                aliases.insert(*target, inputs);
                continue;
            }
            if op.as_str() == "*.elim" {
                let [(source, _)] = assignment.sources.as_slice() else {
                    return Err(CodegenError::InvalidProduct { operation: op });
                };
                let components = self.resolve_node(
                    *source,
                    &aliases,
                    &function_symbols,
                    &term,
                    &instance.substitutions,
                )?;
                let mut offset = 0;
                for (target, _) in &assignment.targets {
                    let count = vars(*target, &term, &instance.substitutions)?.len();
                    let Some(values) = components.get(offset..offset + count) else {
                        return Err(CodegenError::InvalidProduct {
                            operation: op.clone(),
                        });
                    };
                    aliases.insert(*target, values.to_vec());
                    offset += count;
                }
                if offset != components.len() {
                    return Err(CodegenError::InvalidProduct { operation: op });
                }
                continue;
            }
            let inputs = self.resolve_nodes(
                assignment.sources.iter().map(|(node, _)| *node),
                &aliases,
                &function_symbols,
                &term,
                &instance.substitutions,
            )?;
            let mut outputs = Vec::new();
            for (node, _) in &assignment.targets {
                let components = vars(*node, &term, &instance.substitutions)?;
                outputs.extend(components.iter().cloned());
                aliases.insert(*node, components.into_iter().map(GpuValue::Var).collect());
            }
            if inputs.is_empty() && outputs.is_empty() {
                continue;
            }
            let call_symbol = if self.definitions.contains_key(&op) {
                Some(self.ensure_specialization(&op, &inputs, &outputs)?)
            } else {
                None
            };
            assignments.push(GpuAssign {
                op,
                call_symbol,
                inputs,
                outputs,
            });
        }
        let targets = self
            .resolve_nodes(
                term.targets.iter().copied(),
                &aliases,
                &function_symbols,
                &term,
                &instance.substitutions,
            )?
            .into_iter()
            .map(|value| match value {
                GpuValue::Var(var) => Ok(var),
                GpuValue::FnSymbol(_) => {
                    Err(CodegenError::FunctionOutput(instance.operation.clone()))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GpuModule {
            name: instance.symbol.clone(),
            source_name: instance.source_name.clone(),
            entry: GpuFunction {
                name: instance.symbol.clone(),
                sources,
                targets,
                assignments,
            },
        })
    }

    fn ensure_specialization(
        &mut self,
        operation: &Operation,
        inputs: &[GpuValue],
        outputs: &[GpuVar],
    ) -> Result<String, CodegenError> {
        let term = self
            .definitions
            .get(operation)
            .expect("specialized operation should have a definition");
        let mut substitutions = BTreeMap::new();
        infer_boundary(term, inputs, outputs, &mut substitutions, operation)?;
        self.ensure_instance(operation, specialize::key(inputs, outputs), substitutions)
    }

    fn ensure_function_specialization(
        &mut self,
        operation: &Operation,
        ty: &Tree<(), Operation>,
        caller_substitutions: &BTreeMap<usize, CType>,
    ) -> Result<Operation, CodegenError> {
        let (source, target) = function_boundary(ty)
            .ok_or_else(|| CodegenError::InvalidFunctionType(operation.clone()))?;
        let sources = concrete_components(source, caller_substitutions)?;
        let targets = concrete_components(target, caller_substitutions)?;
        let term = self
            .definitions
            .get(operation)
            .ok_or_else(|| CodegenError::InvalidFunctionType(operation.clone()))?;
        let mut substitutions = BTreeMap::new();
        infer_boundary_types(term, &sources, &targets, &mut substitutions, operation)?;
        let symbol =
            self.ensure_instance(operation, boundary_key(sources, targets), substitutions)?;
        Ok(symbol
            .parse()
            .expect("generated specialization symbol should be an operation"))
    }

    fn ensure_instance(
        &mut self,
        operation: &Operation,
        key: SpecializationKey,
        substitutions: BTreeMap<usize, CType>,
    ) -> Result<String, CodegenError> {
        if let Some(symbol) = self.instances.get(&(operation.clone(), key.clone())) {
            return Ok(symbol.clone());
        }
        if key.sources.is_empty() && key.targets.is_empty() && key.static_inputs.is_empty() {
            return Err(CodegenError::NonMonomorphicUse(operation.clone()));
        }
        let symbol = sanitize_ident(&format!(
            "program.{operation}__{}",
            self.next_specialization_id
        ));
        self.next_specialization_id += 1;
        self.instances
            .insert((operation.clone(), key), symbol.clone());
        self.queue.push_back(PendingInstance {
            operation: operation.clone(),
            symbol: symbol.clone(),
            source_name: None,
            substitutions,
        });
        Ok(symbol)
    }

    fn resolve_nodes(
        &mut self,
        nodes: impl IntoIterator<Item = NodeId>,
        aliases: &HashMap<NodeId, Vec<GpuValue>>,
        function_symbols: &HashMap<NodeId, Operation>,
        term: &CodegenTerm,
        substitutions: &BTreeMap<usize, CType>,
    ) -> Result<Vec<GpuValue>, CodegenError> {
        let mut output = Vec::new();
        for node in nodes {
            output.extend(self.resolve_node(
                node,
                aliases,
                function_symbols,
                term,
                substitutions,
            )?);
        }
        Ok(output)
    }

    fn resolve_node(
        &mut self,
        node: NodeId,
        aliases: &HashMap<NodeId, Vec<GpuValue>>,
        function_symbols: &HashMap<NodeId, Operation>,
        term: &CodegenTerm,
        substitutions: &BTreeMap<usize, CType>,
    ) -> Result<Vec<GpuValue>, CodegenError> {
        if let Some(operation) = function_symbols.get(&node) {
            let symbol = self.ensure_function_specialization(
                operation,
                &term.hypergraph.nodes[node.0],
                substitutions,
            )?;
            return Ok(vec![GpuValue::FnSymbol(symbol)]);
        }
        if let Some(values) = aliases.get(&node) {
            return Ok(values.clone());
        }
        Ok(vars(node, term, substitutions)?
            .into_iter()
            .map(GpuValue::Var)
            .collect())
    }
}

fn direct_function_symbols(term: &CodegenTerm) -> Result<HashMap<NodeId, Operation>, CodegenError> {
    let mut symbols = HashMap::new();
    for (edge_index, operation) in term.hypergraph.edges.iter().enumerate() {
        let Some(target) = operation.as_str().strip_prefix("name.") else {
            continue;
        };
        let adjacency = &term.hypergraph.adjacency[edge_index];
        let [node] = adjacency.targets.as_slice() else {
            return Err(CodegenError::InvalidNameArity {
                operation: operation.clone(),
                actual: adjacency.targets.len(),
            });
        };
        let target = target
            .parse()
            .map_err(|_| CodegenError::InvalidNameTarget(target.to_string()))?;
        symbols.insert(*node, target);
    }
    Ok(symbols)
}

fn vars(
    node: NodeId,
    term: &CodegenTerm,
    substitutions: &BTreeMap<usize, CType>,
) -> Result<Vec<GpuVar>, LowerTypeError> {
    let mut output = Vec::new();
    lower_components(
        node,
        &term.hypergraph.nodes[node.0],
        &format!("x{}", node.0),
        substitutions,
        &mut output,
    )?;
    Ok(output)
}

fn lower_components(
    node: NodeId,
    ty: &Tree<(), Operation>,
    name: &str,
    substitutions: &BTreeMap<usize, CType>,
    output: &mut Vec<GpuVar>,
) -> Result<(), LowerTypeError> {
    if let Tree::Node(operation, _, children) = ty
        && operation.as_str() == "*"
    {
        for (index, child) in children.iter().enumerate() {
            lower_components(
                node,
                child,
                &format!("{name}_{index}"),
                substitutions,
                output,
            )?;
        }
        return Ok(());
    }
    let lowered = lower_type(ty, substitutions)?;
    if let LoweredType::Runtime(_) = &lowered {
        output.push(GpuVar {
            node,
            name: name.into(),
            lowered,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum RepresentedComponent<'a> {
    Runtime(&'a Tree<(), Operation>),
    Function,
}

fn represented_components<'a>(
    ty: &'a Tree<(), Operation>,
    output: &mut Vec<RepresentedComponent<'a>>,
) {
    if let Tree::Node(operation, _, children) = ty {
        if operation.as_str() == "*" {
            for child in children {
                represented_components(child, output);
            }
            return;
        }
        if operation.as_str() == "val"
            && matches!(children.as_slice(), [Tree::Node(arrow, _, _)] if arrow.as_str() == "->")
        {
            output.push(RepresentedComponent::Function);
            return;
        }
        if matches!(operation.as_str(), "val" | ":" | "mem") {
            output.push(RepresentedComponent::Runtime(ty));
        }
    }
}

fn concrete_components(
    ty: &Tree<(), Operation>,
    substitutions: &BTreeMap<usize, CType>,
) -> Result<Vec<CType>, LowerTypeError> {
    let mut components = Vec::new();
    represented_components(ty, &mut components);
    components
        .into_iter()
        .filter_map(|component| match component {
            RepresentedComponent::Runtime(ty) => Some(lower_type(ty, substitutions).and_then(
                |lowered| match lowered {
                    LoweredType::Runtime(ty) => Ok(ty),
                    LoweredType::Erased => unreachable!("represented runtime component was erased"),
                },
            )),
            RepresentedComponent::Function => None,
        })
        .collect()
}

fn concrete_boundary(term: &CodegenTerm) -> Result<Option<(Vec<CType>, Vec<CType>)>, CodegenError> {
    let substitutions = BTreeMap::new();
    let result = (|| {
        let mut sources = Vec::new();
        for node in &term.sources {
            sources.extend(concrete_components(
                &term.hypergraph.nodes[node.0],
                &substitutions,
            )?);
        }
        let mut targets = Vec::new();
        for node in &term.targets {
            targets.extend(concrete_components(
                &term.hypergraph.nodes[node.0],
                &substitutions,
            )?);
        }
        Ok::<_, LowerTypeError>((sources, targets))
    })();
    match result {
        Ok((sources, targets)) if !sources.is_empty() || !targets.is_empty() => {
            Ok(Some((sources, targets)))
        }
        Ok(_) | Err(LowerTypeError::Unspecialized(_)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn infer_boundary(
    term: &CodegenTerm,
    inputs: &[GpuValue],
    outputs: &[GpuVar],
    substitutions: &mut BTreeMap<usize, CType>,
    operation: &Operation,
) -> Result<(), CodegenError> {
    let mut formals = Vec::new();
    for node in &term.sources {
        represented_components(&term.hypergraph.nodes[node.0], &mut formals);
    }
    if formals.len() != inputs.len() {
        return Err(CodegenError::SpecializationArity {
            operation: operation.clone(),
            formal: formals.len(),
            actual: inputs.len(),
        });
    }
    for (formal, actual) in formals.into_iter().zip(inputs) {
        match (formal, actual) {
            (RepresentedComponent::Runtime(ty), GpuValue::Var(var)) => {
                infer_type(ty, runtime_type(var).unwrap(), substitutions)?;
            }
            (RepresentedComponent::Function, GpuValue::FnSymbol(_)) => {}
            _ => {
                return Err(CodegenError::NonMonomorphicUse(operation.clone()));
            }
        }
    }

    let target_types = outputs
        .iter()
        .map(|output| runtime_type(output).unwrap().clone())
        .collect::<Vec<_>>();
    infer_boundary_side(
        term.targets
            .iter()
            .map(|node| &term.hypergraph.nodes[node.0]),
        &target_types,
        substitutions,
        operation,
    )
}

fn infer_boundary_types(
    term: &CodegenTerm,
    sources: &[CType],
    targets: &[CType],
    substitutions: &mut BTreeMap<usize, CType>,
    operation: &Operation,
) -> Result<(), CodegenError> {
    infer_boundary_side(
        term.sources
            .iter()
            .map(|node| &term.hypergraph.nodes[node.0]),
        sources,
        substitutions,
        operation,
    )?;
    infer_boundary_side(
        term.targets
            .iter()
            .map(|node| &term.hypergraph.nodes[node.0]),
        targets,
        substitutions,
        operation,
    )
}

fn infer_boundary_side<'a>(
    boundary: impl Iterator<Item = &'a Tree<(), Operation>>,
    concrete: &[CType],
    substitutions: &mut BTreeMap<usize, CType>,
    operation: &Operation,
) -> Result<(), CodegenError> {
    let mut formals = Vec::new();
    for ty in boundary {
        represented_components(ty, &mut formals);
    }
    let runtime_formals = formals
        .into_iter()
        .filter_map(|formal| match formal {
            RepresentedComponent::Runtime(ty) => Some(ty),
            RepresentedComponent::Function => None,
        })
        .collect::<Vec<_>>();
    if runtime_formals.len() != concrete.len() {
        return Err(CodegenError::SpecializationArity {
            operation: operation.clone(),
            formal: runtime_formals.len(),
            actual: concrete.len(),
        });
    }
    for (formal, actual) in runtime_formals.into_iter().zip(concrete) {
        infer_type(formal, actual, substitutions)?;
    }
    Ok(())
}

fn function_boundary(
    ty: &Tree<(), Operation>,
) -> Option<(&Tree<(), Operation>, &Tree<(), Operation>)> {
    let Tree::Node(operation, _, children) = ty else {
        return None;
    };
    match (operation.as_str(), children.as_slice()) {
        ("->", [source, target]) => Some((source, target)),
        ("val", [inner]) => function_boundary(inner),
        (":", [_name, inner]) => function_boundary(inner),
        _ => None,
    }
}

pub fn runtime_type(var: &GpuVar) -> Option<&CType> {
    match &var.lowered {
        LoweredType::Runtime(ty) => Some(ty),
        LoweredType::Erased => None,
    }
}

fn sanitize_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_launch_reachable_through_fold_and_helper_is_rejected() {
        let mut modules = GpuModuleMap::new();
        modules.insert(
            op("entry"),
            module(
                "entry",
                vec![assignment(
                    "gpu.launch",
                    false,
                    vec![GpuValue::FnSymbol(op("outer-kernel"))],
                )],
            ),
        );
        modules.insert(
            op("outer-kernel"),
            module(
                "outer-kernel",
                vec![assignment(
                    "fold",
                    false,
                    vec![GpuValue::FnSymbol(op("fold-step"))],
                )],
            ),
        );
        modules.insert(
            op("fold-step"),
            module("fold-step", vec![assignment("helper", true, Vec::new())]),
        );
        modules.insert(
            op("helper"),
            module("helper", vec![assignment("gpu.launch", false, Vec::new())]),
        );

        let error = validate_device_callability(&modules).unwrap_err();
        assert!(matches!(
            error,
            CodegenError::NestedLaunch { kernel, path }
                if kernel.as_str() == "outer-kernel"
                    && path == "outer-kernel -> fold-step -> helper -> gpu.launch"
        ));
    }

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn module(name: &str, assignments: Vec<GpuAssign>) -> GpuModule {
        GpuModule {
            name: sanitize_ident(&format!("program.{name}")),
            source_name: Some(op(name)),
            entry: GpuFunction {
                name: sanitize_ident(&format!("program.{name}")),
                sources: Vec::new(),
                targets: Vec::new(),
                assignments,
            },
        }
    }

    fn assignment(op_name: &str, direct_call: bool, inputs: Vec<GpuValue>) -> GpuAssign {
        GpuAssign {
            op: op(op_name),
            call_symbol: direct_call.then(|| sanitize_ident(&format!("program.{op_name}"))),
            inputs,
            outputs: Vec::new(),
        }
    }
}
