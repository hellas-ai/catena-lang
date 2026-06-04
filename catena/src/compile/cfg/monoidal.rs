use open_hypergraphs::{
    category::Arrow,
    lax::NodeId,
    strict::vec::{
        FiniteFunction as StrictFiniteFunction, Hypergraph as StrictHypergraph,
        IndexedCoproduct as StrictIndexedCoproduct, OpenHypergraph as StrictOpenHypergraph,
        SemifiniteFunction as StrictSemifiniteFunction, VecArray,
    },
};
use std::collections::HashMap;

use crate::compile::CompileGraph;

use super::{
    model::{CfgError, OperationId, VariableId},
    operation::{
        CONTROL_FLOW_ONLY_OPERATIONS, MONOIDAL_STRUCTURE_OPERATIONS, local_operation_name,
        operation_names, operation_sources, operation_targets,
    },
};

// Monoidal-structure subgraph construction and interpretation

#[derive(Debug, Clone)]
pub(super) struct MonoidalStructureSubgraph {
    graph: StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
}

impl MonoidalStructureSubgraph {
    pub(super) fn from_compile_graph(compile_graph: &CompileGraph) -> Self {
        Self::from_compile_graph_with_context(compile_graph, None, None)
    }

    pub(super) fn from_compile_graph_with_context(
        compile_graph: &CompileGraph,
        wire_map: Option<&std::collections::HashMap<NodeId, VariableId>>,
        inherited: Option<&MonoidalStructureSubgraph>,
    ) -> Self {
        let mut builder = MonoidalStructureSubgraphBuilder::new();
        if let Some(inherited) = inherited {
            builder.add_subgraph(inherited);
        }
        builder.add_region(compile_graph, wire_map);
        Self {
            graph: builder.finish(),
        }
    }
}

pub(super) struct MonoidalStructureSubgraphBuilder {
    wires: Vec<crate::lang::Obj>,
    operations: Vec<crate::lang::Arr>,
    source_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    source_values: Vec<usize>,
    target_values: Vec<usize>,
}

impl MonoidalStructureSubgraphBuilder {
    fn new() -> Self {
        Self {
            wires: Vec::new(),
            operations: Vec::new(),
            source_lengths: Vec::new(),
            target_lengths: Vec::new(),
            source_values: Vec::new(),
            target_values: Vec::new(),
        }
    }

    fn add_subgraph(&mut self, subgraph: &MonoidalStructureSubgraph) {
        for (wire, object) in subgraph.graph.h.w.0.0.iter().cloned().enumerate() {
            self.add_wire(wire, object);
        }
        for operation_id in 0..subgraph_operation_count(&subgraph.graph) {
            self.add_operation(
                subgraph.graph.h.x.0.0[operation_id].clone(),
                monoidal_structure_operation_sources(&subgraph.graph, operation_id)
                    .into_iter()
                    .map(|wire| wire.0)
                    .collect(),
                monoidal_structure_operation_targets(&subgraph.graph, operation_id)
                    .into_iter()
                    .map(|wire| wire.0)
                    .collect(),
            );
        }
    }

    fn add_region(
        &mut self,
        compile_graph: &CompileGraph,
        wire_map: Option<&std::collections::HashMap<NodeId, VariableId>>,
    ) {
        for (wire, object) in compile_graph.graph.h.w.0.0.iter().cloned().enumerate() {
            self.add_wire(mapped_region_wire(NodeId(wire), wire_map), object);
        }
        for operation_id in 0..operation_names(compile_graph).len() {
            let operation_name = operation_names(compile_graph)[operation_id].to_string();
            if is_structure_resolver_operation(local_operation_name(&operation_name)) {
                self.add_operation(
                    operation_names(compile_graph)[operation_id].clone(),
                    operation_sources(compile_graph, operation_id)
                        .into_iter()
                        .map(|wire| mapped_region_wire(wire, wire_map))
                        .collect(),
                    operation_targets(compile_graph, operation_id)
                        .into_iter()
                        .map(|wire| mapped_region_wire(wire, wire_map))
                        .collect(),
                );
            }
        }
    }

    fn add_wire(&mut self, wire: VariableId, object: crate::lang::Obj) {
        if self.wires.len() <= wire {
            self.wires.resize(wire + 1, object.clone());
        }
        self.wires[wire] = object;
    }

    fn add_operation(
        &mut self,
        operation: crate::lang::Arr,
        sources: Vec<usize>,
        targets: Vec<usize>,
    ) {
        self.operations.push(operation);
        self.source_lengths.push(sources.len());
        self.target_lengths.push(targets.len());
        self.source_values.extend(sources);
        self.target_values.extend(targets);
    }

    fn finish(self) -> StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr> {
        let wire_count = self.wires.len();
        StrictOpenHypergraph {
            s: StrictFiniteFunction::identity(wire_count),
            t: StrictFiniteFunction::identity(wire_count),
            h: StrictHypergraph {
                s: indexed_coproduct(self.source_lengths, self.source_values, wire_count),
                t: indexed_coproduct(self.target_lengths, self.target_values, wire_count),
                w: StrictSemifiniteFunction::new(VecArray(self.wires)),
                x: StrictSemifiniteFunction::new(VecArray(self.operations)),
            },
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct MonoidalStructureResolver<'a> {
    compile_graph: &'a CompileGraph,
    subgraph: MonoidalStructureSubgraph,
    values: HashMap<VariableId, MonoidalValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MonoidalValue {
    Atom(VariableId),
    Product {
        origin: Option<VariableId>,
        components: Vec<MonoidalValue>,
    },
    Coproduct {
        condition: Box<MonoidalValue>,
        branches: Vec<MonoidalValue>,
    },
    Unit,
}

#[allow(dead_code)]
impl<'a> MonoidalStructureResolver<'a> {
    pub(super) fn new(compile_graph: &'a CompileGraph) -> Self {
        Self::new_with_context(compile_graph, None, None)
    }

    pub(super) fn new_with_context(
        compile_graph: &'a CompileGraph,
        wire_map: Option<&std::collections::HashMap<NodeId, VariableId>>,
        inherited: Option<&MonoidalStructureSubgraph>,
    ) -> Self {
        let subgraph = MonoidalStructureSubgraph::from_compile_graph_with_context(
            compile_graph,
            wire_map,
            inherited,
        );
        let values = interpret_monoidal_structure(&subgraph);
        Self {
            compile_graph,
            subgraph,
            values,
        }
    }

    pub(super) fn from_subgraph(
        compile_graph: &'a CompileGraph,
        subgraph: MonoidalStructureSubgraph,
    ) -> Self {
        let values = interpret_monoidal_structure(&subgraph);
        Self {
            compile_graph,
            subgraph,
            values,
        }
    }

    pub(super) fn subgraph(&self) -> &MonoidalStructureSubgraph {
        &self.subgraph
    }

    pub(super) fn resolve_variables(
        &self,
        variables: Vec<VariableId>,
    ) -> Result<Vec<VariableId>, CfgError> {
        variables
            .into_iter()
            .map(|variable| self.resolve_variable(variable))
            .collect()
    }

    pub(super) fn is_structure_wire(&self, variable: VariableId) -> bool {
        producer_of_monoidal_structure_wire(&self.subgraph.graph, variable).is_some()
    }

    pub(super) fn resolve_discriminator(
        &self,
        variable: VariableId,
    ) -> Result<VariableId, CfgError> {
        match self
            .values
            .get(&variable)
            .cloned()
            .unwrap_or(MonoidalValue::Atom(variable))
        {
            MonoidalValue::Atom(atom) => Ok(atom),
            MonoidalValue::Coproduct { condition, .. } => {
                atom_value(*condition).ok_or_else(|| CfgError::UnresolvedMonoidalStructureAtom {
                    wire: variable,
                    operation: "coproduct discriminator".to_string(),
                })
            }
            value => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation: format!("{value:?}"),
            }),
        }
    }

    pub(super) fn resolve_branch_payload_wire(&self, variable: VariableId) -> VariableId {
        match self.values.get(&variable).cloned() {
            Some(MonoidalValue::Product { mut components, .. }) if components.len() > 1 => {
                origin_wire(components.remove(1))
                    .or_else(|| self.branch_payload_from_producer(variable))
                    .unwrap_or(variable)
            }
            Some(value) => origin_wire(value)
                .or_else(|| self.branch_payload_from_producer(variable))
                .unwrap_or(variable),
            None => self
                .branch_payload_from_producer(variable)
                .unwrap_or(variable),
        }
    }

    fn branch_payload_from_producer(&self, variable: VariableId) -> Option<VariableId> {
        let (operation_id, branch) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)?;
        let operation = monoidal_structure_operation_name(&self.subgraph.graph, operation_id);
        match operation.as_str() {
            "val.+.elim" => {
                let source = single_source(&monoidal_structure_operation_sources(
                    &self.subgraph.graph,
                    operation_id,
                ))?;
                let branch = coproduct_branches(source, &self.values)?
                    .get(branch)?
                    .clone();
                product_components_from_value(branch)
                    .and_then(|mut components| (components.len() > 1).then(|| components.remove(1)))
                    .and_then(origin_wire)
            }
            "elim2" => origin_wire(value_of(variable, &self.values)),
            _ => None,
        }
    }

    fn resolve_variable(&self, variable: VariableId) -> Result<VariableId, CfgError> {
        match self
            .values
            .get(&variable)
            .cloned()
            .unwrap_or(MonoidalValue::Atom(variable))
        {
            MonoidalValue::Atom(atom) => Ok(atom),
            value => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation: format!("{value:?}"),
            }),
        }
    }
}

fn interpret_monoidal_structure(
    subgraph: &MonoidalStructureSubgraph,
) -> HashMap<VariableId, MonoidalValue> {
    let mut values = (0..subgraph.graph.h.w.0.len())
        .map(|wire| (wire, MonoidalValue::Atom(wire)))
        .collect::<HashMap<_, _>>();

    for operation_id in 0..subgraph_operation_count(&subgraph.graph) {
        let operation = monoidal_structure_operation_name(&subgraph.graph, operation_id);
        let sources = monoidal_structure_operation_sources(&subgraph.graph, operation_id);
        let targets = monoidal_structure_operation_targets(&subgraph.graph, operation_id);
        interpret_monoidal_operation(&operation, &sources, &targets, &mut values);
    }

    values
}

fn interpret_monoidal_operation(
    operation: &str,
    sources: &[NodeId],
    targets: &[NodeId],
    values: &mut HashMap<VariableId, MonoidalValue>,
) {
    match operation {
        "val.*.intro" => {
            if let Some(target) = single_target(targets) {
                values.insert(
                    target,
                    MonoidalValue::Product {
                        origin: Some(target),
                        components: source_values(sources, values),
                    },
                );
            }
        }
        "val.*.elim" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some(components) = product_components(source, values) {
                assign_targets(targets, components, values);
            }
        }
        "unitl.intro" => {
            if let (Some(source), Some(target)) = (single_source(sources), single_target(targets)) {
                values.insert(
                    target,
                    MonoidalValue::Product {
                        origin: Some(target),
                        components: vec![MonoidalValue::Unit, value_of(source, values)],
                    },
                );
            }
        }
        "unitl.elim" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some(component) = product_components(source, values).and_then(|mut values| {
                if values.len() > 1 {
                    Some(values.remove(1))
                } else {
                    None
                }
            }) {
                assign_targets(targets, vec![component], values);
            }
        }
        "2.elim" => {
            if let (Some(source), Some(target)) = (single_source(sources), single_target(targets)) {
                values.insert(
                    target,
                    MonoidalValue::Coproduct {
                        condition: Box::new(value_of(source, values)),
                        branches: vec![MonoidalValue::Unit, MonoidalValue::Unit],
                    },
                );
            }
        }
        "2.intro" => {
            if let (Some(source), Some(target)) = (single_source(sources), single_target(targets)) {
                values.insert(target, value_of(source, values));
            }
        }
        "val.+.intro" => {
            if let Some(target) = single_target(targets) {
                values.insert(
                    target,
                    MonoidalValue::Coproduct {
                        condition: Box::new(MonoidalValue::Unit),
                        branches: source_values(sources, values),
                    },
                );
            }
        }
        "val.+.elim" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some(branches) = coproduct_branches(source, values) {
                assign_targets(targets, branches, values);
            }
        }
        "distr" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some([left, right]) = product_pair(source, values)
                && let MonoidalValue::Coproduct {
                    condition,
                    branches,
                } = left
            {
                let distributed = branches
                    .into_iter()
                    .map(|branch| MonoidalValue::Product {
                        origin: None,
                        components: vec![branch, right.clone()],
                    })
                    .collect();
                assign_targets(
                    targets,
                    vec![MonoidalValue::Coproduct {
                        condition,
                        branches: distributed,
                    }],
                    values,
                );
            }
        }
        "distl" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some([left, right]) = product_pair(source, values)
                && let MonoidalValue::Coproduct {
                    condition,
                    branches,
                } = right
            {
                let distributed = branches
                    .into_iter()
                    .map(|branch| MonoidalValue::Product {
                        origin: None,
                        components: vec![left.clone(), branch],
                    })
                    .collect();
                assign_targets(
                    targets,
                    vec![MonoidalValue::Coproduct {
                        condition,
                        branches: distributed,
                    }],
                    values,
                );
            }
        }
        "elim2" => {
            let Some(source) = single_source(sources) else {
                return;
            };
            if let Some(branches) = coproduct_branches(source, values) {
                let eliminated = branches
                    .into_iter()
                    .filter_map(|branch| match branch {
                        MonoidalValue::Product { mut components, .. } if components.len() > 1 => {
                            Some(components.remove(1))
                        }
                        _ => None,
                    })
                    .collect();
                assign_targets(targets, eliminated, values);
            }
        }
        "merge" => {
            if let Some(target) = single_target(targets) {
                values.insert(
                    target,
                    MonoidalValue::Coproduct {
                        condition: Box::new(MonoidalValue::Unit),
                        branches: source_values(sources, values),
                    },
                );
            }
        }
        _ => {}
    }
}

fn single_source(sources: &[NodeId]) -> Option<VariableId> {
    let [source] = sources else {
        return None;
    };
    Some(source.0)
}

fn single_target(targets: &[NodeId]) -> Option<VariableId> {
    let [target] = targets else {
        return None;
    };
    Some(target.0)
}

fn value_of(wire: VariableId, values: &HashMap<VariableId, MonoidalValue>) -> MonoidalValue {
    values
        .get(&wire)
        .cloned()
        .unwrap_or(MonoidalValue::Atom(wire))
}

fn atom_value(value: MonoidalValue) -> Option<VariableId> {
    match value {
        MonoidalValue::Atom(atom) => Some(atom),
        _ => None,
    }
}

fn origin_wire(value: MonoidalValue) -> Option<VariableId> {
    match value {
        MonoidalValue::Atom(atom) => Some(atom),
        MonoidalValue::Product { origin, .. } => origin,
        MonoidalValue::Coproduct { .. } | MonoidalValue::Unit => None,
    }
}

fn source_values(
    sources: &[NodeId],
    values: &HashMap<VariableId, MonoidalValue>,
) -> Vec<MonoidalValue> {
    sources
        .iter()
        .map(|source| value_of(source.0, values))
        .collect()
}

fn assign_targets(
    targets: &[NodeId],
    assigned: Vec<MonoidalValue>,
    values: &mut HashMap<VariableId, MonoidalValue>,
) {
    for (target, value) in targets.iter().zip(assigned) {
        values.insert(target.0, value);
    }
}

fn product_components(
    wire: VariableId,
    values: &HashMap<VariableId, MonoidalValue>,
) -> Option<Vec<MonoidalValue>> {
    product_components_from_value(value_of(wire, values))
}

fn product_components_from_value(value: MonoidalValue) -> Option<Vec<MonoidalValue>> {
    match value {
        MonoidalValue::Product { components, .. } => Some(components),
        _ => None,
    }
}

fn coproduct_branches(
    wire: VariableId,
    values: &HashMap<VariableId, MonoidalValue>,
) -> Option<Vec<MonoidalValue>> {
    match value_of(wire, values) {
        MonoidalValue::Coproduct { branches, .. } => Some(branches),
        _ => None,
    }
}

fn product_pair(
    wire: VariableId,
    values: &HashMap<VariableId, MonoidalValue>,
) -> Option<[MonoidalValue; 2]> {
    let components = product_components(wire, values)?;
    let [left, right]: [MonoidalValue; 2] = components.try_into().ok()?;
    Some([left, right])
}

fn is_structure_resolver_operation(operation: &str) -> bool {
    MONOIDAL_STRUCTURE_OPERATIONS.contains(&operation)
        || CONTROL_FLOW_ONLY_OPERATIONS
            .iter()
            .any(|control_operation| {
                *control_operation == "merge" && operation == *control_operation
            })
}

fn mapped_region_wire(
    wire: NodeId,
    wire_map: Option<&std::collections::HashMap<NodeId, VariableId>>,
) -> VariableId {
    wire_map
        .and_then(|wire_map| wire_map.get(&wire).copied())
        .unwrap_or(wire.0)
}

fn indexed_coproduct(
    segment_lengths: Vec<usize>,
    values: Vec<usize>,
    target: usize,
) -> StrictIndexedCoproduct<StrictFiniteFunction> {
    let total = segment_lengths.iter().sum::<usize>();
    debug_assert_eq!(total, values.len());
    let sources = StrictFiniteFunction::new(VecArray(segment_lengths), total + 1)
        .expect("monoidal-structure subgraph segment lengths must form a valid indexed coproduct");
    let values = StrictFiniteFunction::new(VecArray(values), target)
        .expect("monoidal-structure subgraph incidence values must reference existing wires");
    StrictIndexedCoproduct::new(sources, values)
        .expect("monoidal-structure subgraph incidence must be valid")
}

fn subgraph_operation_count(
    subgraph: &StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
) -> usize {
    subgraph.h.x.0.len()
}

#[allow(dead_code)]
fn producer_of_monoidal_structure_wire(
    subgraph: &StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
    wire: VariableId,
) -> Option<(OperationId, usize)> {
    (0..subgraph_operation_count(subgraph)).find_map(|operation_id| {
        monoidal_structure_operation_targets(subgraph, operation_id)
            .iter()
            .position(|target| target.0 == wire)
            .map(|output_index| (operation_id, output_index))
    })
}

fn monoidal_structure_operation_sources(
    subgraph: &StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
    operation_id: OperationId,
) -> Vec<NodeId> {
    subgraph
        .h
        .s
        .clone()
        .into_iter()
        .nth(operation_id)
        .map(|sources| sources.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
}

#[allow(dead_code)]
fn monoidal_structure_operation_name(
    subgraph: &StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
    operation_id: OperationId,
) -> String {
    local_operation_name(&subgraph.h.x.0[operation_id].to_string()).to_string()
}

fn monoidal_structure_operation_targets(
    subgraph: &StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
    operation_id: OperationId,
) -> Vec<NodeId> {
    subgraph
        .h
        .t
        .clone()
        .into_iter()
        .nth(operation_id)
        .map(|targets| targets.table.0.into_iter().map(NodeId).collect())
        .unwrap_or_default()
}
