use open_hypergraphs::{
    category::Arrow,
    lax::NodeId,
    strict::vec::{
        FiniteFunction as StrictFiniteFunction, Hypergraph as StrictHypergraph,
        IndexedCoproduct as StrictIndexedCoproduct, OpenHypergraph as StrictOpenHypergraph,
        SemifiniteFunction as StrictSemifiniteFunction, VecArray,
    },
};
use std::collections::HashSet;

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
        Self {
            compile_graph,
            subgraph: MonoidalStructureSubgraph::from_compile_graph_with_context(
                compile_graph,
                wire_map,
                inherited,
            ),
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
        if !seen.insert(variable) {
            return Err(CfgError::MonoidalStructureCycle(variable));
        }

        let Some((operation_id, output_index)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return Ok(variable);
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
            "merge" => {
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
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
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
        let Some((operation_id, output_index)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation: "product component".to_string(),
            });
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
            "merge" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                self.resolve_product_component(source.0, component, seen)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn resolve_coproduct_branch_atom(
        &self,
        variable: VariableId,
        branch: usize,
        _seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let Some((operation_id, _)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation: "coproduct branch".to_string(),
            });
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
            "merge" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                self.resolve_coproduct_branch_atom(source.0, branch, _seen)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }

    fn resolve_coproduct_branch_product_component(
        &self,
        variable: VariableId,
        branch: usize,
        component: usize,
        seen: &mut HashSet<VariableId>,
    ) -> Result<VariableId, CfgError> {
        let Some((operation_id, _)) =
            producer_of_monoidal_structure_wire(&self.subgraph.graph, variable)
        else {
            return Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation: "coproduct branch".to_string(),
            });
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
            "merge" => {
                let sources =
                    monoidal_structure_operation_sources(&self.subgraph.graph, operation_id);
                let [source] = sources.as_slice() else {
                    return Err(CfgError::UnresolvedMonoidalStructureAtom {
                        wire: variable,
                        operation,
                    });
                };
                self.resolve_coproduct_branch_product_component(source.0, branch, component, seen)
            }
            _ => Err(CfgError::UnresolvedMonoidalStructureAtom {
                wire: variable,
                operation,
            }),
        }
    }
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
