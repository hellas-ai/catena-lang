use std::collections::HashMap;

use open_hypergraphs::{
    lax::{NodeId, OpenHypergraph as LaxOpenHypergraph, functor::Functor},
    strict::vec::FiniteFunction,
};

use crate::{
    compile::{
        CompileGraph, CompileTheory,
        analysis::{
            control_regions::ControlRegionGraph,
            layering::{
                BoundaryRelation, NestedGraph, coproduct_over_parent, quotient_over_parent,
            },
            partition::{OperationId, OperationRegion, RegionKind, SourceOperation},
            wires::is_interleaved_data_operation,
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    pass::inline::Inline,
    stdlib::operations::actual_operation_name,
};

#[derive(Debug, Clone)]
pub struct DataRegionGraph {
    pub region_index: usize,
    pub source_operations: Vec<SourceOperation>,
    pub nested_graph: NestedGraph,
    pub regions: Vec<OperationRegion>,
    pub control_region_graphs: Vec<ControlRegionGraph>,
}

pub(super) fn process_data_regions(
    definition_context: &CompileGraph,
    parent_graph: &Graph,
    regions: &[OperationRegion],
) -> Vec<DataRegionGraph> {
    regions
        .iter()
        .enumerate()
        .filter(|(_, region)| matches!(region.kind, RegionKind::InterleavedData))
        .map(|(region_index, region)| {
            expand_data_region(definition_context, parent_graph, region_index, region)
        })
        .collect()
}

// Expand one interleaved data region from a control graph into a native data
// graph. Definitions are inlined recursively; primitive cross-theory children
// are already represented as one-operation compile graphs by graph building.
fn expand_data_region(
    definition_context: &CompileGraph,
    parent_graph: &Graph,
    region_index: usize,
    region: &OperationRegion,
) -> DataRegionGraph {
    let nested_graph = coproduct_over_parent(
        region
            .operations
            .iter()
            .copied()
            .map(|operation_id| {
                expand_interleaved_data_call(definition_context, parent_graph, operation_id)
            })
            .collect::<Vec<_>>(),
    );
    let resolved = quotient_over_parent(parent_graph, nested_graph);

    DataRegionGraph {
        region_index,
        source_operations: source_operations(parent_graph, region),
        nested_graph: resolved,
        regions: Vec::new(),
        control_region_graphs: Vec::new(),
    }
}

fn expand_interleaved_data_call(
    definition_context: &CompileGraph,
    parent_graph: &Graph,
    operation_id: OperationId,
) -> NestedGraph {
    debug_assert!(is_interleaved_data_operation(parent_graph, operation_id));
    let operation = operation_name(parent_graph, operation_id);
    let expanded_data_graph = if let Some(native_data_child) =
        data_definition_for_operation(definition_context, operation)
    {
        inline_data_definitions(native_data_child)
    } else {
        primitive_data_graph(parent_graph, operation_id)
    };

    let expanded_source_wires = boundary_table(&expanded_data_graph.s);
    let expanded_target_wires = boundary_table(&expanded_data_graph.t);
    let call_inputs = operation_inputs(parent_graph, operation_id).collect::<Vec<_>>();
    let call_outputs = operation_outputs(parent_graph, operation_id).collect::<Vec<_>>();
    let boundary_relation = BoundaryRelation::from_boundaries(
        operation_id,
        (&expanded_source_wires, &call_inputs),
        (&expanded_target_wires, &call_outputs),
    );

    NestedGraph::under_parent_operation(operation_id, expanded_data_graph, boundary_relation)
}

fn inline_data_definitions(graph: &CompileGraph) -> Graph {
    debug_assert!(matches!(graph.theory, CompileTheory::Data));
    let definitions = graph
        .children
        .iter()
        .filter(|child| matches!(child.graph.theory, CompileTheory::Data))
        .map(|child| {
            (
                child
                    .operation
                    .parse()
                    .expect("data child operation name must be valid"),
                LaxOpenHypergraph::from_strict(inline_data_definitions(&child.graph)),
            )
        })
        .collect::<HashMap<_, _>>();
    if definitions.is_empty() {
        return graph.graph.clone();
    }

    let mut expanded = LaxOpenHypergraph::from_strict(graph.graph.clone());
    for _ in 0..64 {
        let inlinable = expanded
            .hypergraph
            .edges
            .iter()
            .any(|operation| definitions.contains_key(operation));
        if !inlinable {
            return expanded.to_strict();
        }
        expanded = Inline {
            definitions: definitions.clone(),
        }
        .map_arrow(&expanded);
    }

    panic!(
        "too many data-definition inline iterations while expanding `{}`",
        graph.definition_name
    )
}

fn data_definition_for_operation<'a>(
    graph: &'a CompileGraph,
    operation: &str,
) -> Option<&'a CompileGraph> {
    graph.children.iter().find_map(|child| {
        if child.operation == operation && matches!(child.graph.theory, CompileTheory::Data) {
            Some(&child.graph)
        } else {
            data_definition_for_operation(&child.graph, operation)
        }
    })
}

fn primitive_data_graph(parent_graph: &Graph, operation_id: OperationId) -> Graph {
    let operation = operation_name(parent_graph, operation_id);
    let native_operation = actual_operation_name(operation)
        .parse()
        .expect("interleaved data operation name must strip to a valid operation");
    let source_type = operation_inputs(parent_graph, operation_id)
        .map(|wire| parent_graph.h.w.0.0[wire.0].clone())
        .collect::<Vec<_>>();
    let target_type = operation_outputs(parent_graph, operation_id)
        .map(|wire| parent_graph.h.w.0.0[wire.0].clone())
        .collect::<Vec<_>>();

    LaxOpenHypergraph::singleton(native_operation, source_type, target_type).to_strict()
}

fn boundary_table(boundary: &FiniteFunction) -> Vec<NodeId> {
    boundary.table.0.iter().copied().map(NodeId).collect()
}

fn source_operations(parent_graph: &Graph, region: &OperationRegion) -> Vec<SourceOperation> {
    region
        .operations
        .iter()
        .copied()
        .map(|id| SourceOperation {
            id,
            name: operation_name(parent_graph, id).to_string(),
        })
        .collect()
}
