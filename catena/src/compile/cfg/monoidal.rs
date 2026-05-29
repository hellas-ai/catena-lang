use open_hypergraphs::{
    category::Arrow,
    lax::NodeId,
    strict::vec::{
        FiniteFunction as StrictFiniteFunction, Hypergraph as StrictHypergraph,
        IndexedCoproduct as StrictIndexedCoproduct, OpenHypergraph as StrictOpenHypergraph,
        SemifiniteFunction as StrictSemifiniteFunction, VecArray,
    },
};
use std::collections::{HashMap, HashSet};

use crate::compile::CompileGraph;

use super::{
    model::{CfgError, OperationId, VariableId},
    operation::{
        CfgOperationRole, MONOIDAL_STRUCTURE_OPERATIONS, cfg_operation_role,
        child_graph_for_operation, is_control_operation, local_operation_name, operation_names,
        operation_sources, operation_targets, source_nodes, target_nodes,
    },
};

// Monoidal-structure subgraph construction and interpretation

#[derive(Debug, Clone)]
pub(super) struct MonoidalStructureSubgraph {
    graph: StrictOpenHypergraph<crate::lang::Obj, crate::lang::Arr>,
}

impl MonoidalStructureSubgraph {
    pub(super) fn from_compile_graph(compile_graph: &CompileGraph) -> Self {
        let mut builder = MonoidalStructureSubgraphBuilder::new(compile_graph.graph.h.w.0.to_vec());
        builder.add_region(compile_graph);
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
    fn new(wires: Vec<crate::lang::Obj>) -> Self {
        Self {
            wires,
            operations: Vec::new(),
            source_lengths: Vec::new(),
            target_lengths: Vec::new(),
            source_values: Vec::new(),
            target_values: Vec::new(),
        }
    }

    fn add_region(&mut self, compile_graph: &CompileGraph) {
        for operation_id in 0..operation_names(compile_graph).len() {
            let operation_name = operation_names(compile_graph)[operation_id].to_string();
            if MONOIDAL_STRUCTURE_OPERATIONS.contains(&local_operation_name(&operation_name)) {
                self.add_operation(
                    operation_names(compile_graph)[operation_id].clone(),
                    operation_sources(compile_graph, operation_id)
                        .into_iter()
                        .map(|wire| wire.0)
                        .collect(),
                    operation_targets(compile_graph, operation_id)
                        .into_iter()
                        .map(|wire| wire.0)
                        .collect(),
                );
            }

            if should_compose_monoidal_structure_child(compile_graph, &operation_name)
                && let Some(child) = child_graph_for_operation(compile_graph, &operation_name)
            {
                self.add_child_region(compile_graph, operation_id, child);
            }
        }
    }

    fn add_child_region(
        &mut self,
        parent: &CompileGraph,
        operation_id: OperationId,
        child: &CompileGraph,
    ) {
        let child_subgraph = MonoidalStructureSubgraph::from_compile_graph(child);
        let mut wire_map = HashMap::<usize, usize>::new();
        for (child_wire, parent_wire) in source_nodes(child)
            .into_iter()
            .zip(operation_sources(parent, operation_id))
        {
            wire_map.insert(child_wire.0, parent_wire.0);
        }
        for (child_wire, parent_wire) in target_nodes(child)
            .into_iter()
            .zip(operation_targets(parent, operation_id))
        {
            wire_map.insert(child_wire.0, parent_wire.0);
        }

        for child_wire in 0..child_subgraph.graph.h.w.0.len() {
            wire_map.entry(child_wire).or_insert_with(|| {
                let id = self.wires.len();
                self.wires
                    .push(child_subgraph.graph.h.w.0[child_wire].clone());
                id
            });
        }

        for child_operation_id in 0..subgraph_operation_count(&child_subgraph.graph) {
            self.add_operation(
                child_subgraph.graph.h.x.0[child_operation_id].clone(),
                monoidal_structure_operation_sources(&child_subgraph.graph, child_operation_id)
                    .into_iter()
                    .map(|wire| wire_map[&wire.0])
                    .collect(),
                monoidal_structure_operation_targets(&child_subgraph.graph, child_operation_id)
                    .into_iter()
                    .map(|wire| wire_map[&wire.0])
                    .collect(),
            );
        }
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

fn should_compose_monoidal_structure_child(compile_graph: &CompileGraph, operation: &str) -> bool {
    matches!(
        cfg_operation_role(operation),
        CfgOperationRole::MonoidalStructure | CfgOperationRole::ControlFlow
    ) || is_control_operation(compile_graph, operation)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) struct MonoidalStructureResolver {
    subgraph: MonoidalStructureSubgraph,
    wire_map: HashMap<VariableId, VariableId>,
    inverse_wire_map: HashMap<VariableId, VariableId>,
    fallback_wires: HashSet<VariableId>,
    fallback: Option<Box<MonoidalStructureResolver>>,
}

#[allow(dead_code)]
impl MonoidalStructureResolver {
    pub(super) fn new(subgraph: MonoidalStructureSubgraph) -> Self {
        Self {
            subgraph,
            wire_map: HashMap::new(),
            inverse_wire_map: HashMap::new(),
            fallback_wires: HashSet::new(),
            fallback: None,
        }
    }

    pub(super) fn child_resolver(
        &self,
        subgraph: MonoidalStructureSubgraph,
        wire_map: HashMap<VariableId, VariableId>,
        fallback_wires: HashSet<VariableId>,
    ) -> Self {
        let inverse_wire_map = wire_map
            .iter()
            .map(|(local, mapped)| (*mapped, *local))
            .collect();
        Self {
            subgraph,
            wire_map,
            inverse_wire_map,
            fallback_wires,
            fallback: Some(Box::new(self.clone())),
        }
    }

    pub(super) fn resolve_variables(
        &self,
        variables: Vec<VariableId>,
    ) -> Result<Vec<VariableId>, CfgError> {
        variables
            .into_iter()
            .map(|variable| self.resolve_atom(variable))
            .collect()
    }

    pub(super) fn resolve_atom(&self, variable: VariableId) -> Result<VariableId, CfgError> {
        self.resolve_atom_inner(variable, &mut HashSet::new())
    }

    fn resolve_atom_inner(
        &self,
        variable: VariableId,
        seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let variable = self.local_wire(variable);
        if !seen.insert(variable) {
            return Err(CfgError::MonoidalStructureCycle(variable));
        }

        let Some((operation_id, output_index)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return self.resolve_external_atom(variable);
        };

        let operation = monoidal_structure_operation_name(&self.subgraph.graph, operation_id);
        match operation.as_str() {
            "val.*.elim" => {
                self.resolve_val_product_elim_component(operation_id, output_index, seen)
            }
            "2.elim" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                self.resolve_atom_inner(source.0, seen)
            }
            "distr" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let condition = self.resolve_product_component(source.0, 0, seen)?;
                self.resolve_atom_inner(condition, seen)
            }
            "unitl.elim" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let payload = self.resolve_product_component(source.0, 1, seen)?;
                self.resolve_atom_inner(payload, seen)
            }
            "val.+.elim" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let branch = self.resolve_coproduct_branch_atom(source.0, output_index, seen)?;
                self.resolve_atom_inner(branch, seen)
            }
            "elim2" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let payload = self.resolve_coproduct_branch_product_component(
                    source.0,
                    output_index,
                    1,
                    seen,
                )?;
                self.resolve_atom_inner(payload, seen)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn local_wire(&self, variable: VariableId) -> VariableId {
        self.inverse_wire_map
            .get(&variable)
            .copied()
            .unwrap_or(variable)
    }

    fn resolve_external_atom(&self, variable: VariableId) -> Result<VariableId, CfgError> {
        let mapped = self.wire_map.get(&variable).copied().unwrap_or(variable);
        if self.fallback_wires.contains(&variable)
            && let Some(fallback) = &self.fallback
        {
            fallback.resolve_atom(mapped)
        } else {
            Ok(mapped)
        }
    }

    fn resolve_val_product_elim_component(
        &self,
        elim_operation: OperationId,
        component: usize,
        seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let packed_sources =
            monoidal_structure_operation_sources(&self.subgraph.graph, elim_operation);
        let [packed] = packed_sources.as_slice() else {
            return Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: usize::MAX,
                operation: monoidal_structure_operation_name(&self.subgraph.graph, elim_operation),
            });
        };
        let source = self.resolve_product_component(packed.0, component, seen)?;
        self.resolve_atom_inner(source, seen)
    }

    fn resolve_product_component(
        &self,
        variable: VariableId,
        component: usize,
        seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let variable = self.local_wire(variable);
        let Some((operation_id, output_index)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return self.resolve_external_product_component(variable, component);
        };

        let operation = monoidal_structure_operation_name(&self.subgraph.graph, operation_id);
        match operation.as_str() {
            "val.*.intro" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                sources
                    .get(component)
                    .map(|source| source.0)
                    .ok_or_else(|| CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    })
            }
            "val.+.elim" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                self.resolve_coproduct_branch_product_component(
                    source.0,
                    output_index,
                    component,
                    seen,
                )
            }
            "distr" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let payload = self.resolve_product_component(source.0, 1, seen)?;
                self.resolve_product_component(payload, component, seen)
            }
            "unitl.intro" => {
                if component != 1 {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                }
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                Ok(source.0)
            }
            "elim2" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                let payload = self.resolve_coproduct_branch_product_component(
                    source.0,
                    output_index,
                    1,
                    seen,
                )?;
                self.resolve_product_component(payload, component, seen)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn resolve_external_product_component(
        &self,
        variable: VariableId,
        component: usize,
    ) -> Result<VariableId, CfgError> {
        if self.fallback_wires.contains(&variable)
            && let Some(fallback) = &self.fallback
        {
            let mapped = self.wire_map.get(&variable).copied().unwrap_or(variable);
            return fallback.resolve_product_component(mapped, component, &mut HashSet::new());
        }

        Err(CfgError::UnresolvedMonoidalStructureAtom {
            wire: self.wire_map.get(&variable).copied().unwrap_or(variable),
            operation: "product component".to_string(),
        })
    }

    fn resolve_coproduct_branch_atom(
        &self,
        variable: VariableId,
        branch: usize,
        _seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let variable = self.local_wire(variable);
        let Some((operation_id, _)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return self.resolve_external_coproduct_branch_atom(variable, branch);
        };

        let operation = monoidal_structure_operation_name(&self.subgraph.graph, operation_id);
        match operation.as_str() {
            "val.+.intro" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                sources.get(branch).map(|source| source.0).ok_or_else(|| {
                    CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    }
                })
            }
            "2.elim" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                Ok(source.0)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn resolve_external_coproduct_branch_atom(
        &self,
        variable: VariableId,
        branch: usize,
    ) -> Result<VariableId, CfgError> {
        if self.fallback_wires.contains(&variable)
            && let Some(fallback) = &self.fallback
        {
            let mapped = self.wire_map.get(&variable).copied().unwrap_or(variable);
            return fallback.resolve_coproduct_branch_atom(mapped, branch, &mut HashSet::new());
        }

        Err(CfgError::UnresolvedMonoidalStructureAtom {
            wire: self.wire_map.get(&variable).copied().unwrap_or(variable),
            operation: "coproduct branch".to_string(),
        })
    }

    fn resolve_coproduct_branch_product_component(
        &self,
        variable: VariableId,
        branch: usize,
        component: usize,
        seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let variable = self.local_wire(variable);
        let Some((operation_id, _)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return self
                .resolve_external_coproduct_branch_product_component(variable, branch, component);
        };
        let operation = monoidal_structure_operation_name(&self.subgraph.graph, operation_id);
        match operation.as_str() {
            "val.+.intro" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let source = sources.get(branch).ok_or_else(|| {
                    CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    }
                })?;
                self.resolve_product_component(source.0, component, seen)
            }
            "distr" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                match component {
                    0 => {
                        let coproduct = self.resolve_product_component(source.0, 0, seen)?;
                        self.resolve_coproduct_branch_atom(coproduct, branch, seen)
                    }
                    1 => self.resolve_product_component(source.0, 1, seen),
                    _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    }),
                }
            }
            "distl" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                match component {
                    0 => self.resolve_product_component(source.0, 0, seen),
                    1 => {
                        let coproduct = self.resolve_product_component(source.0, 1, seen)?;
                        self.resolve_coproduct_branch_atom(coproduct, branch, seen)
                    }
                    _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    }),
                }
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn resolve_external_coproduct_branch_product_component(
        &self,
        variable: VariableId,
        branch: usize,
        component: usize,
    ) -> Result<VariableId, CfgError> {
        if self.fallback_wires.contains(&variable)
            && let Some(fallback) = &self.fallback
        {
            let mapped = self.wire_map.get(&variable).copied().unwrap_or(variable);
            return fallback.resolve_coproduct_branch_product_component(
                mapped,
                branch,
                component,
                &mut HashSet::new(),
            );
        }

        Err(CfgError::UnresolvedMonoidalStructureAtom {
            wire: self.wire_map.get(&variable).copied().unwrap_or(variable),
            operation: "coproduct branch".to_string(),
        })
    }
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
