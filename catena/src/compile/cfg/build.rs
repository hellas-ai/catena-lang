use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::compile::{CompileGraph, CompileTheory};

use super::{
    control::{ControlExpander, ExpandedControlItem},
    data::{
        block_instructions, control_region_block_instructions, data_cfg_node_draft,
        partition_data_operations_by_internal_wires,
    },
    model::{
        BoundaryKind, Cfg, CfgEdge, CfgError, CfgNode, CfgNodeDraft, CfgNodeId, CfgWiring,
        OperationId, VariableId,
    },
    monoidal::{MonoidalStructureResolver, MonoidalStructureSubgraph},
    normalize::normalize_cfg,
    operation::{
        CfgOperationRole, OperationInstance, cfg_operation_role, effective_operation_instance,
        is_branch_operation, is_control_operation, operation_names,
    },
    wiring::{
        BoundaryWires, cfg_node_from_control_draft, data_transfer, nodes_with_boundary,
        remap_transfer_targets, resolve_nested_data_return,
    },
};

// CFG construction

#[derive(Debug)]
pub(super) struct CfgBuilder<'a> {
    compile_graph: &'a CompileGraph,
    wire_map: HashMap<NodeId, VariableId>,
    monoidal_structure_resolver: MonoidalStructureResolver<'a>,
    node_ids: CfgNodeIdAllocator,
    operation_instances: Vec<OperationInstance>,
    control_operation_ids: Vec<OperationId>,
    data_operation_ids: Vec<OperationId>,
}

impl<'a> CfgBuilder<'a> {
    pub(super) fn new(compile_graph: &'a CompileGraph) -> Self {
        Self::new_with_context(compile_graph, HashMap::new())
    }

    pub(super) fn new_with_context(
        compile_graph: &'a CompileGraph,
        wire_map: HashMap<NodeId, VariableId>,
    ) -> Self {
        Self::new_with_context_and_monoidal(compile_graph, wire_map, None)
    }

    pub(super) fn new_with_context_and_monoidal(
        compile_graph: &'a CompileGraph,
        wire_map: HashMap<NodeId, VariableId>,
        inherited_monoidal_structure: Option<MonoidalStructureSubgraph>,
    ) -> Self {
        let monoidal_structure_resolver = MonoidalStructureResolver::new_with_context(
            compile_graph,
            Some(&wire_map),
            inherited_monoidal_structure.as_ref(),
        );
        Self {
            compile_graph,
            wire_map,
            monoidal_structure_resolver,
            node_ids: CfgNodeIdAllocator::default(),
            operation_instances: Vec::new(),
            control_operation_ids: Vec::new(),
            data_operation_ids: Vec::new(),
        }
    }

    pub(super) fn build(mut self) -> Result<Cfg, CfgError> {
        self.collect_operations()?;
        if matches!(self.compile_graph.theory, CompileTheory::Data) {
            self.build_data_cfg()
        } else if matches!(self.compile_graph.theory, CompileTheory::Control) {
            self.build_control_cfg()
        } else {
            Err(CfgError::UnsupportedTheory(
                self.compile_graph.theory.clone(),
            ))
        }
    }

    fn collect_operations(&mut self) -> Result<(), CfgError> {
        self.operation_instances = (0..operation_names(self.compile_graph).len())
            .map(|operation_id| {
                effective_operation_instance(
                    self.compile_graph,
                    operation_id,
                    &self.wire_map,
                    &self.monoidal_structure_resolver,
                )
            })
            .collect::<Result<Vec<_>, CfgError>>()?;

        for operation in &self.operation_instances {
            match cfg_runtime_role(self.compile_graph, operation) {
                Some(CfgOperationRole::ControlFlow) => {
                    self.control_operation_ids.push(operation.id)
                }
                Some(CfgOperationRole::Instruction) => self.data_operation_ids.push(operation.id),
                Some(CfgOperationRole::MonoidalStructure) | None => {}
            }
        }
        Ok(())
    }

    fn build_data_cfg(&mut self) -> Result<Cfg, CfgError> {
        let control_fragment = self.control_cfg_fragment()?;
        let control_operations = control_fragment
            .control_operation_by_node
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let boundary = BoundaryWires::from_region_and_control_operations(
            self.compile_graph,
            &control_operations,
            &self.wire_map,
            &self.monoidal_structure_resolver,
        );
        let data_fragment = self.data_cfg_fragment(&boundary)?;
        Ok(self.compose_fragments(data_fragment, control_fragment))
    }

    fn build_control_cfg(&mut self) -> Result<Cfg, CfgError> {
        let control_fragment = self.control_cfg_fragment()?;
        Ok(self.compose_top_level_control(control_fragment))
    }

    fn control_cfg_fragment(&mut self) -> Result<ControlCfgFragment, CfgError> {
        let expanded_control = ControlExpander::new(
            self.compile_graph,
            &self.operation_instances,
            self.monoidal_structure_resolver.subgraph().clone(),
        )
        .expand(&self.control_operation_ids)?;

        let mut node_by_control_operation = HashMap::new();
        let mut control_operation_by_node = HashMap::new();
        let mut node_by_entry_wire = HashMap::new();
        let mut nested_data_nodes = Vec::new();
        let mut nested_data_node_by_entry_wire = HashMap::new();
        let mut branch_data_successors = HashMap::<OperationId, Vec<CfgEdge>>::new();
        let mut current_branch = None::<OperationInstance>;
        let mut nodes = Vec::new();

        for item in expanded_control.items {
            match item {
                ExpandedControlItem::Control(operation) => {
                    let id = self.node_ids.allocate();
                    node_by_control_operation.insert(operation.id, id);
                    control_operation_by_node.insert(id, operation.clone());
                    for input in &operation.inputs {
                        node_by_entry_wire.insert(*input, id);
                    }
                    nodes.push(CfgNodeDraft {
                        id,
                        params: operation.inputs.clone(),
                        block: if matches!(self.compile_graph.theory, CompileTheory::Control) {
                            control_region_block_instructions(operation)?
                        } else {
                            block_instructions(operation)?
                        },
                    });
                    current_branch = control_operation_by_node
                        .get(&id)
                        .filter(|operation| is_branch_operation(operation))
                        .cloned();
                }
                ExpandedControlItem::DataCfg { call, cfg } => {
                    let remapped_cfg = self.remap_cfg_nodes(cfg);
                    if let Some(entry) = remapped_cfg
                        .nodes
                        .iter()
                        .find(|node| node.id == remapped_cfg.entry)
                    {
                        for input in &call.inputs {
                            nested_data_node_by_entry_wire.insert(*input, entry.id);
                        }
                        if let Some(branch) = &current_branch {
                            let successors = branch_data_successors.entry(branch.id).or_default();
                            let arg = if entry.params.is_empty() {
                                branch
                                    .outputs
                                    .get(successors.len())
                                    .copied()
                                    .or_else(|| call.inputs.first().copied())
                                    .into_iter()
                                    .collect()
                            } else {
                                entry.params.clone()
                            };
                            successors.push(CfgEdge {
                                target: entry.id,
                                args: arg,
                            });
                        }
                    }
                    nested_data_nodes.extend(remapped_cfg.nodes);
                }
            }
        }

        for (visible_operation, entry_operation) in expanded_control.visible_operation_to_entry {
            if let Some(entry_node) = node_by_control_operation.get(&entry_operation).copied() {
                node_by_control_operation.insert(visible_operation, entry_node);
            }
        }
        Ok(ControlCfgFragment {
            nodes,
            nested_data_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_data_successors,
        })
    }

    fn remap_cfg_nodes(&mut self, mut cfg: Cfg) -> Cfg {
        let node_id_by_old = cfg
            .nodes
            .iter()
            .map(|node| (node.id, self.node_ids.allocate()))
            .collect::<HashMap<_, _>>();

        for node in &mut cfg.nodes {
            node.id = node_id_by_old[&node.id];
            node.transfer = remap_transfer_targets(node.transfer.clone(), &node_id_by_old);
        }
        cfg.entry = node_id_by_old[&cfg.entry];
        cfg
    }

    fn data_cfg_fragment(&mut self, boundary: &BoundaryWires) -> Result<DataCfgFragment, CfgError> {
        let operations_by_cfg_node = partition_data_operations_by_internal_wires(
            self.compile_graph,
            &self.operation_instances,
            &self.data_operation_ids,
            &boundary.all,
        );
        let mut node_by_entry_wire = HashMap::new();
        let mut node_boundaries = Vec::new();

        let mut nodes = Vec::new();
        for operations in operations_by_cfg_node {
            let id = self.node_ids.allocate();
            let (node, boundaries) =
                data_cfg_node_draft(self.compile_graph, id, operations, boundary)?;

            for point in &boundaries.entries {
                node_by_entry_wire.insert(point.wire, id);
            }

            node_boundaries.push(boundaries);
            nodes.push(node);
        }

        Ok(DataCfgFragment {
            nodes,
            wiring: CfgWiring { node_boundaries },
            node_by_entry_wire,
        })
    }

    fn compose_fragments(
        &self,
        data_fragment: DataCfgFragment,
        control_fragment: ControlCfgFragment,
    ) -> Cfg {
        let DataCfgFragment {
            nodes: data_nodes,
            wiring,
            node_by_entry_wire: data_node_by_entry_wire,
        } = data_fragment;
        let ControlCfgFragment {
            nodes: control_nodes,
            nested_data_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire: control_node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_data_successors,
        } = control_fragment;
        let mut data_node_by_entry_wire = data_node_by_entry_wire;
        data_node_by_entry_wire.extend(nested_data_node_by_entry_wire);
        let preludes = nested_data_prelude_entries(&nested_data_nodes, &control_node_by_entry_wire);
        data_node_by_entry_wire.extend(preludes);
        let mut synthetic_return_nodes = Vec::new();
        let mut next_synthetic_node = control_nodes
            .iter()
            .map(|node| node.id)
            .chain(nested_data_nodes.iter().map(|node| node.id))
            .chain(data_nodes.iter().map(|node| node.id))
            .max()
            .map(|id| id + 1)
            .unwrap_or(0);
        for operation in control_operation_by_node.values() {
            if !is_branch_operation(operation) {
                continue;
            }
            for (index, output) in operation.outputs.iter().copied().enumerate() {
                let has_branch_data_successor = branch_data_successors
                    .get(&operation.id)
                    .is_some_and(|successors| successors.get(index).is_some());
                if has_branch_data_successor
                    || control_node_by_entry_wire.contains_key(&output)
                    || data_node_by_entry_wire.contains_key(&output)
                {
                    continue;
                }
                let id = next_synthetic_node;
                next_synthetic_node += 1;
                data_node_by_entry_wire.insert(output, id);
                synthetic_return_nodes.push(CfgNode {
                    id,
                    params: vec![output],
                    block: Vec::new(),
                    transfer: super::model::Transfer::Return(vec![output]),
                });
            }
        }

        let boundaries_by_node = wiring
            .node_boundaries
            .iter()
            .map(|boundaries| (boundaries.node, boundaries))
            .collect::<HashMap<_, _>>();

        let mut nodes = control_nodes
            .into_iter()
            .map(|node| {
                cfg_node_from_control_draft(
                    node,
                    &control_operation_by_node,
                    &control_node_by_entry_wire,
                    &data_node_by_entry_wire,
                    &branch_data_successors,
                )
            })
            .collect::<Vec<_>>();
        nodes.extend(nested_data_nodes.into_iter().map(|mut node| {
            node.transfer = resolve_nested_data_return(
                node.transfer,
                &control_node_by_entry_wire,
                &data_node_by_entry_wire,
            );
            node
        }));
        nodes.extend(synthetic_return_nodes);
        nodes.extend(data_nodes.into_iter().map(|node| {
            let boundaries = boundaries_by_node
                .get(&node.id)
                .expect("data node must have boundary wiring");
            CfgNode {
                id: node.id,
                params: node.params,
                block: node.block,
                transfer: data_transfer(boundaries, &node_by_control_operation),
            }
        }));
        let entry = cfg_entry_from_region_entries(&nodes, &wiring);
        normalize_cfg(Cfg {
            entry,
            nodes,
            predecessors: Vec::new(),
        })
    }

    fn compose_top_level_control(&self, control_fragment: ControlCfgFragment) -> Cfg {
        let ControlCfgFragment {
            nodes: control_nodes,
            nested_data_nodes,
            node_by_control_operation: _,
            control_operation_by_node,
            node_by_entry_wire: control_node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_data_successors,
        } = control_fragment;
        let data_node_by_entry_wire = nested_data_node_by_entry_wire;

        let mut nodes = control_nodes
            .into_iter()
            .map(|node| {
                let operation = control_operation_by_node
                    .get(&node.id)
                    .expect("control node must have source operation");
                let transfer = top_level_control_transfer(
                    node.id,
                    operation,
                    &control_node_by_entry_wire,
                    &data_node_by_entry_wire,
                    &branch_data_successors,
                    self.compile_graph,
                    &self.wire_map,
                );
                CfgNode {
                    id: node.id,
                    params: node.params,
                    block: node.block,
                    transfer,
                }
            })
            .collect::<Vec<_>>();

        nodes.extend(nested_data_nodes.into_iter().map(|mut node| {
            node.transfer = resolve_nested_data_return(
                node.transfer,
                &control_node_by_entry_wire,
                &data_node_by_entry_wire,
            );
            node
        }));

        let region_sources = crate::compile::cfg::operation::source_nodes(self.compile_graph)
            .into_iter()
            .filter_map(|wire| self.wire_map.get(&wire).copied().or(Some(wire.0)))
            .collect::<std::collections::HashSet<_>>();
        let entry = nodes
            .iter()
            .find(|node| {
                node.params
                    .iter()
                    .any(|param| region_sources.contains(param))
            })
            .map(|node| node.id)
            .or_else(|| nodes.first().map(|node| node.id))
            .unwrap_or(0);
        normalize_cfg(Cfg {
            entry,
            nodes,
            predecessors: Vec::new(),
        })
    }
}

fn cfg_runtime_role(
    compile_graph: &CompileGraph,
    operation: &OperationInstance,
) -> Option<CfgOperationRole> {
    match cfg_operation_role(&operation.name) {
        CfgOperationRole::MonoidalStructure => None,
        CfgOperationRole::ControlFlow => Some(CfgOperationRole::ControlFlow),
        CfgOperationRole::Instruction if matches!(compile_graph.theory, CompileTheory::Control) => {
            Some(CfgOperationRole::ControlFlow)
        }
        CfgOperationRole::Instruction if is_control_operation(compile_graph, &operation.name) => {
            Some(CfgOperationRole::ControlFlow)
        }
        CfgOperationRole::Instruction => Some(CfgOperationRole::Instruction),
    }
}

fn nested_data_prelude_entries(
    nested_data_nodes: &[CfgNode],
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
) -> HashMap<VariableId, CfgNodeId> {
    let mut entries = HashMap::new();
    for node in nested_data_nodes {
        let target = match &node.transfer {
            super::model::Transfer::Goto(edge) => Some(edge.target),
            super::model::Transfer::Return(values) => values
                .iter()
                .find_map(|value| control_node_by_entry_wire.get(value).copied()),
            super::model::Transfer::If { .. } => None,
        };
        let Some(target) = target else {
            continue;
        };
        for (wire, control_node) in control_node_by_entry_wire {
            if *control_node == target {
                entries.insert(*wire, node.id);
            }
        }
    }
    entries
}

fn top_level_control_transfer(
    node: CfgNodeId,
    operation: &OperationInstance,
    control_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    data_node_by_entry_wire: &HashMap<VariableId, CfgNodeId>,
    branch_data_successors: &HashMap<OperationId, Vec<CfgEdge>>,
    compile_graph: &CompileGraph,
    wire_map: &HashMap<NodeId, VariableId>,
) -> super::model::Transfer {
    let transfer = crate::compile::cfg::wiring::control_transfer(
        node,
        operation,
        control_node_by_entry_wire,
        data_node_by_entry_wire,
        branch_data_successors,
    );
    if !matches!(transfer, super::model::Transfer::Return(_)) {
        return transfer;
    }

    let region_targets = crate::compile::cfg::operation::target_nodes(compile_graph)
        .into_iter()
        .map(|wire| wire_map.get(&wire).copied().unwrap_or(wire.0))
        .collect::<std::collections::HashSet<_>>();
    let returns = operation
        .outputs
        .iter()
        .copied()
        .filter(|wire| region_targets.contains(wire))
        .collect::<Vec<_>>();
    super::model::Transfer::Return(returns)
}

fn cfg_entry_from_region_entries(nodes: &[CfgNode], wiring: &CfgWiring) -> CfgNodeId {
    let region_entries = nodes_with_boundary(wiring, BoundaryKind::RegionEntry);
    region_entries
        .iter()
        .copied()
        .find(|entry| {
            nodes
                .iter()
                .find(|node| node.id == *entry)
                .is_some_and(|node| !node.block.is_empty())
        })
        .or_else(|| region_entries.into_iter().next())
        .or_else(|| nodes.first().map(|node| node.id))
        .unwrap_or(0)
}

// CFG construction state

#[derive(Debug, Default)]
pub(super) struct CfgNodeIdAllocator {
    next: CfgNodeId,
}

#[derive(Debug)]
pub(super) struct OperationIdAllocator {
    next: OperationId,
}

impl OperationIdAllocator {
    pub(super) fn new(next: OperationId) -> Self {
        Self { next }
    }

    pub(super) fn allocate(&mut self) -> OperationId {
        let id = self.next;
        self.next += 1;
        id
    }
}

#[derive(Debug)]
pub(super) struct VariableIdAllocator {
    next: VariableId,
}

impl VariableIdAllocator {
    pub(super) fn new(next: VariableId) -> Self {
        Self { next }
    }

    pub(super) fn allocate(&mut self) -> VariableId {
        let id = self.next;
        self.next += 1;
        id
    }
}

impl CfgNodeIdAllocator {
    pub(super) fn allocate(&mut self) -> CfgNodeId {
        let id = self.next;
        self.next += 1;
        id
    }
}

#[derive(Debug, Clone)]
pub(super) struct DataCfgFragment {
    nodes: Vec<CfgNodeDraft>,
    wiring: CfgWiring,
    node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
}

#[derive(Debug, Clone)]
pub(super) struct ControlCfgFragment {
    nodes: Vec<CfgNodeDraft>,
    nested_data_nodes: Vec<CfgNode>,
    node_by_control_operation: HashMap<OperationId, CfgNodeId>,
    control_operation_by_node: HashMap<CfgNodeId, OperationInstance>,
    node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    nested_data_node_by_entry_wire: HashMap<VariableId, CfgNodeId>,
    branch_data_successors: HashMap<OperationId, Vec<CfgEdge>>,
}
