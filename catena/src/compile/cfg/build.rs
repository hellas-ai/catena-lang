use std::collections::{HashMap, HashSet};

use open_hypergraphs::lax::NodeId;

use crate::compile::{CompileGraph, CompileTheory};

use super::{
    control::{ControlExpander, ExpandedControlItem},
    data::{block_instructions, data_cfg_node_draft, partition_data_operations_by_internal_wires},
    model::{
        Cfg, CfgEdge, CfgError, CfgNode, CfgNodeDraft, CfgNodeId, CfgOptions, CfgWiring,
        OperationId, VariableId,
    },
    monoidal::{MonoidalStructureResolver, MonoidalStructureSubgraph},
    operation::{
        CfgOperationRole, OperationInstance, cfg_operation_role, effective_operation_instance,
        is_branch_operation, is_control_operation, local_operation_name, operation_names,
    },
    wiring::{
        BoundaryWires, cfg_node_from_control_draft, data_transfer, nodes_with_boundary,
        predecessors, remap_transfer_targets, resolve_nested_data_return,
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
    options: CfgOptions,
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
            options: CfgOptions::default(),
        }
    }

    pub(super) fn with_options(mut self, options: CfgOptions) -> Self {
        self.options = options;
        self
    }

    pub(super) fn build(mut self) -> Result<Cfg, CfgError> {
        self.reject_non_data_region()?;
        self.collect_operations()?;
        self.build_data_cfg()
    }

    fn reject_non_data_region(&self) -> Result<(), CfgError> {
        match &self.compile_graph.theory {
            CompileTheory::Data => Ok(()),
            other => Err(CfgError::UnsupportedTheory(other.clone())),
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
                    self.options,
                )
            })
            .collect::<Result<Vec<_>, CfgError>>()?;

        for operation in &self.operation_instances {
            // Temporary hack: `control.elim2` is a monoidal definition, so it must
            // remain visible to the monoidal resolver, but expanding it as a CFG
            // control node breaks branch wiring. Remove this once monoidal
            // definitions are handled before control expansion.
            if operation.name.starts_with("control.")
                && local_operation_name(&operation.name) == "elim2"
            {
                continue;
            }
            if is_control_operation(self.compile_graph, &operation.name) {
                self.control_operation_ids.push(operation.id);
            } else {
                self.data_operation_ids.push(operation.id);
            }
        }
        Ok(())
    }

    fn build_data_cfg(&mut self) -> Result<Cfg, CfgError> {
        let boundary = BoundaryWires::from_region_and_control_operations(
            self.compile_graph,
            &self.operation_instances,
            &self.control_operation_ids,
            &self.wire_map,
        );

        let control_fragment = self.control_cfg_fragment()?;
        let data_fragment = self.data_cfg_fragment(&boundary)?;
        Ok(self.compose_fragments(data_fragment, control_fragment))
    }

    fn control_cfg_fragment(&mut self) -> Result<ControlCfgFragment, CfgError> {
        let expanded_control = ControlExpander::new(
            self.compile_graph,
            &self.operation_instances,
            self.monoidal_structure_resolver.subgraph().clone(),
            self.options,
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
                        block: block_instructions(operation, self.options)?,
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
                            let arg = branch
                                .outputs
                                .get(successors.len())
                                .copied()
                                .or_else(|| call.inputs.first().copied())
                                .into_iter()
                                .collect();
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
                data_cfg_node_draft(self.compile_graph, id, operations, boundary, self.options)?;
            if node.block.is_empty() && boundaries.exits.is_empty() {
                continue;
            }

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
        let erased_monoidal_wires = self.erased_monoidal_wires(control_operation_by_node.values());

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
        for node in &mut nodes {
            if !self.options.keep_monoidal_operations {
                erase_monoidal_structure_params_and_args(
                    node,
                    &self.monoidal_structure_resolver,
                    &erased_monoidal_wires,
                );
            }
            prune_unused_params(node);
        }
        truncate_edge_args_to_target_params(&mut nodes);
        let entry = nodes_with_boundary(&wiring, super::model::BoundaryKind::RegionEntry)
            .into_iter()
            .next()
            .or_else(|| nodes.first().map(|node| node.id))
            .unwrap_or(0);
        nodes.sort_by_key(|node| node.id);
        let predecessors = predecessors(&nodes);

        Cfg {
            entry,
            nodes,
            predecessors,
        }
    }

    fn erased_monoidal_wires<'b>(
        &self,
        control_operations: impl Iterator<Item = &'b OperationInstance>,
    ) -> HashSet<VariableId> {
        let mut wires = self
            .operation_instances
            .iter()
            .filter(|operation| {
                cfg_operation_role(&operation.name) == CfgOperationRole::MonoidalStructure
            })
            .flat_map(|operation| operation.outputs.iter().copied())
            .collect::<HashSet<_>>();
        wires.extend(
            control_operations
                .filter(|operation| {
                    cfg_operation_role(&operation.name) == CfgOperationRole::MonoidalStructure
                })
                .flat_map(|operation| operation.outputs.iter().copied()),
        );
        wires
    }
}

fn prune_unused_params(node: &mut CfgNode) {
    let mut used = node
        .block
        .iter()
        .flat_map(|instruction| instruction.args.iter().copied())
        .collect::<std::collections::HashSet<_>>();
    match &node.transfer {
        super::model::Transfer::Goto(edge) => {
            used.extend(edge.args.iter().copied());
        }
        super::model::Transfer::If {
            condition,
            then_edge,
            else_edge,
        } => {
            used.insert(*condition);
            used.extend(then_edge.args.iter().copied());
            used.extend(else_edge.args.iter().copied());
        }
        super::model::Transfer::Return(values) => {
            used.extend(values.iter().copied());
        }
    }
    node.params.retain(|param| used.contains(param));
}

fn erase_monoidal_structure_params_and_args(
    node: &mut CfgNode,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    erased_monoidal_wires: &HashSet<VariableId>,
) {
    node.params.retain(|param| {
        !is_erased_monoidal_wire(*param, monoidal_structure_resolver, erased_monoidal_wires)
    });
    match &mut node.transfer {
        super::model::Transfer::Goto(edge) => {
            edge.args.retain(|arg| {
                !is_erased_monoidal_wire(*arg, monoidal_structure_resolver, erased_monoidal_wires)
            });
        }
        super::model::Transfer::If {
            then_edge,
            else_edge,
            ..
        } => {
            then_edge.args.retain(|arg| {
                !is_erased_monoidal_wire(*arg, monoidal_structure_resolver, erased_monoidal_wires)
            });
            else_edge.args.retain(|arg| {
                !is_erased_monoidal_wire(*arg, monoidal_structure_resolver, erased_monoidal_wires)
            });
        }
        super::model::Transfer::Return(_) => {}
    }
}

fn is_erased_monoidal_wire(
    variable: VariableId,
    monoidal_structure_resolver: &MonoidalStructureResolver<'_>,
    erased_monoidal_wires: &HashSet<VariableId>,
) -> bool {
    erased_monoidal_wires.contains(&variable)
        || monoidal_structure_resolver.is_structure_wire(variable)
}

fn truncate_edge_args_to_target_params(nodes: &mut [CfgNode]) {
    let param_counts = nodes
        .iter()
        .map(|node| (node.id, node.params.len()))
        .collect::<HashMap<_, _>>();
    for node in nodes {
        match &mut node.transfer {
            super::model::Transfer::Goto(edge) => truncate_edge_args(edge, &param_counts),
            super::model::Transfer::If {
                then_edge,
                else_edge,
                ..
            } => {
                truncate_edge_args(then_edge, &param_counts);
                truncate_edge_args(else_edge, &param_counts);
            }
            super::model::Transfer::Return(_) => {}
        }
    }
}

fn truncate_edge_args(edge: &mut CfgEdge, param_counts: &HashMap<CfgNodeId, usize>) {
    if let Some(param_count) = param_counts.get(&edge.target) {
        edge.args.truncate(*param_count);
    }
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
