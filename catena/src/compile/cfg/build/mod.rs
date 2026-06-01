use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;

use crate::compile::{CompileGraph, CompileTheory};

mod compose;
mod control;
mod data;
mod monoidal;
mod normalize;
mod operation;

use self::{
    compose::{
        BranchTargets, ControlPlan, DataPlan, compose_control_region, compose_data_region,
        remap_transfer_targets,
    },
    control::{ControlExpander, ExpandedControlItem},
    data::{
        DataBoundaries, block_instructions, control_region_block_instructions, data_cfg_node_draft,
        partition_data_operations_by_internal_wires,
    },
    monoidal::{MonoidalStructureResolver, MonoidalStructureSubgraph},
    operation::PreparedOperations,
};
use super::model::{
    Cfg, CfgEdge, CfgError, CfgNodeDraft, CfgNodeId, CfgWiring, OperationId, VariableId,
};

// CFG construction pipeline:
//
// The input is an open hypergraph whose operations mix:
// - data instructions that become statements inside basic blocks,
// - control operations that become CFG nodes/transfers,
// - monoidal-structure operations that only repack wires.
//
// We do not lower the hypergraph operation-by-operation. Instead we first interpret the monoidal subgraph, then build a CFG from the operations that still have CFG meaning:
//
//   CompileGraph
//     -> PreparedOperations resolve wires and split data/control operations
//     -> ControlPlan       expand control children and nested data calls
//     -> DataBoundaries    decide where data blocks start/end
//     -> DataPlan          partition data operations into CFG block drafts
//     -> compose           connect control nodes and data blocks
//     -> normalize         erase mechanical artifacts and compact node ids
//
// The key idea is that wires, not syntax, decide CFG edges. A control output wire is a boundary for the data block that consumes it, and a data output wire is a boundary for the control node that consumes it.

#[derive(Debug)]
pub(super) struct CfgBuilder<'a> {
    compile_graph: &'a CompileGraph,
    wire_map: HashMap<NodeId, VariableId>,
    monoidal_structure_resolver: MonoidalStructureResolver<'a>,
    node_ids: CfgNodeIdAllocator,
    operations: PreparedOperations,
}

impl<'a> CfgBuilder<'a> {
    pub(super) fn new(compile_graph: &'a CompileGraph) -> Result<Self, CfgError> {
        Self::new_with_context(compile_graph, HashMap::new())
    }

    fn new_with_context(
        compile_graph: &'a CompileGraph,
        wire_map: HashMap<NodeId, VariableId>,
    ) -> Result<Self, CfgError> {
        Self::new_with_context_and_monoidal(compile_graph, wire_map, None)
    }

    fn new_with_context_and_monoidal(
        compile_graph: &'a CompileGraph,
        wire_map: HashMap<NodeId, VariableId>,
        inherited_monoidal_structure: Option<MonoidalStructureSubgraph>,
    ) -> Result<Self, CfgError> {
        // Build the resolver before collecting CFG operations. The resolver interprets structural wiring such as:
        //
        //   x y --val.*.intro--> p --val.*.elim--> u v
        //
        // as the atom mapping:
        //
        //   u -> x
        //   v -> y
        //
        let monoidal_structure_resolver = MonoidalStructureResolver::new_with_context(
            compile_graph,
            Some(&wire_map),
            inherited_monoidal_structure.as_ref(),
        );
        let operations =
            PreparedOperations::collect(compile_graph, &wire_map, &monoidal_structure_resolver)?;
        Ok(Self {
            compile_graph,
            wire_map,
            monoidal_structure_resolver,
            node_ids: CfgNodeIdAllocator::default(),
            operations,
        })
    }

    pub(super) fn build(mut self) -> Result<Cfg, CfgError> {
        match self.compile_graph.theory {
            CompileTheory::Data => self.build_data_cfg(),
            CompileTheory::Control => self.build_control_cfg(),
        }
    }

    fn build_data_cfg(&mut self) -> Result<Cfg, CfgError> {
        // A data region may contain embedded control calls. We first expand those calls into a control skeleton, then use the skeleton's wires as boundaries for partitioning the remaining data operations.
        //
        // Hypergraph shape:
        //
        //   data ops -> if/control -> data ops
        //
        // CFG shape:
        //
        //   [data block] -> [control node] -> [data block]
        let control_plan = self.control_cfg_fragment()?;
        let control_operations = control_plan
            .control_operation_by_node
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let boundary = DataBoundaries::from_region_and_control_operations(
            self.compile_graph,
            &control_operations,
            &self.wire_map,
            &self.monoidal_structure_resolver,
        );
        let data_plan = self.data_cfg_fragment(&boundary)?;
        Ok(compose_data_region(data_plan, control_plan))
    }

    fn build_control_cfg(&mut self) -> Result<Cfg, CfgError> {
        // A top-level control region has no surrounding data partition. We only expand its control skeleton and any nested data CFGs, then compose those pieces directly.
        let control_plan = self.control_cfg_fragment()?;
        Ok(compose_control_region(
            control_plan,
            self.compile_graph,
            &self.wire_map,
        ))
    }

    fn control_cfg_fragment(&mut self) -> Result<ControlPlan, CfgError> {
        // Expand control operations before building final CFG nodes.
        //
        // Control children are inlined:
        //
        //   control.foo(...)
        //
        // becomes the operations inside foo's control graph.
        //
        // Data children are compiled recursively and kept as nested CFGs:
        //
        //   if ----branch 0----> data child CFG
        //      `---branch 1----> data child CFG
        //
        // At this phase we allocate fresh node ids for all control drafts and remap nested CFG node ids so every node id is unique in the eventual composed CFG.
        let expanded_control = ControlExpander::new(
            self.compile_graph,
            &self.operations.operations,
            self.monoidal_structure_resolver.subgraph().clone(),
        )
        .expand(&self.operations.control_ids)?;

        let mut node_by_control_operation = HashMap::new();
        let mut control_operation_by_node = HashMap::new();
        let mut node_by_entry_wire = HashMap::new();
        let mut nested_data_nodes = Vec::new();
        let mut nested_data_node_by_entry_wire = HashMap::new();
        let mut branch_targets = BranchTargets::new();
        let mut nodes = Vec::new();

        for item in expanded_control.items {
            match item {
                ExpandedControlItem::Control(operation) => {
                    // A control operation becomes a CFG node draft. Its inputs are node params because a predecessor must pass those wires along the eventual CFG edge.
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
                }
                ExpandedControlItem::DataCfg {
                    branch_arm,
                    call,
                    cfg,
                } => {
                    // A nested data child has already been compiled into a CFG.
                    // We remap its node ids into this builder's id space, then remember which wires can enter that nested CFG.
                    //
                    // If the data child is a branch arm, also record the edge from the branch node to this data CFG entry:
                    //
                    //        [branch]
                    //        /     \
                    //   arm 0       arm 1
                    //    edge        edge
                    //     |           |
                    // [data cfg]  [data cfg]
                    let remapped_cfg = self.remap_cfg_nodes(cfg);
                    if let Some(entry) = remapped_cfg
                        .nodes
                        .iter()
                        .find(|node| node.id == remapped_cfg.entry)
                    {
                        for input in &call.inputs {
                            nested_data_node_by_entry_wire.insert(*input, entry.id);
                        }
                        if let Some(branch_arm) = branch_arm {
                            let successors =
                                branch_targets.entry(branch_arm.branch.id).or_default();
                            let arg = if entry.params.is_empty() {
                                branch_arm
                                    .branch
                                    .outputs
                                    .get(branch_arm.index)
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
        Ok(ControlPlan {
            nodes,
            nested_data_nodes,
            node_by_control_operation,
            control_operation_by_node,
            node_by_entry_wire,
            nested_data_node_by_entry_wire,
            branch_targets,
        })
    }

    fn remap_cfg_nodes(&mut self, mut cfg: Cfg) -> Cfg {
        // Nested data CFGs are built independently, so their local node ids may collide with ids already allocated in this builder. Remapping turns:
        //
        //   nested: n0 -> n1
        //
        // into fresh ids in the parent CFG:
        //
        //   parent: n7 -> n8
        //
        // and updates every transfer target plus the nested entry id.
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

    fn data_cfg_fragment(&mut self, boundary: &DataBoundaries) -> Result<DataPlan, CfgError> {
        // Partition data operations by internal wires. Operations connected only through non-boundary wires stay in the same data block:
        //
        //   a -> op1 -> w -> op2 -> b       w is internal
        //
        // becomes one block:
        //
        //   [op1; op2]
        //
        // If a wire is a region/control boundary, it cuts the block:
        //
        //   control output -> op1 -> control input
        //
        // becomes a data node with explicit entry/exit boundary metadata.
        let operations_by_cfg_node = partition_data_operations_by_internal_wires(
            self.compile_graph,
            &self.operations.operations,
            &self.operations.data_ids,
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

        Ok(DataPlan {
            nodes,
            wiring: CfgWiring { node_boundaries },
            node_by_entry_wire,
        })
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
