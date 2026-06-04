use std::collections::HashMap;

use open_hypergraphs::{
    lax::{NodeId, OpenHypergraph as LaxOpenHypergraph, functor::Functor},
    strict::vec::{
        FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph, SemifiniteFunction, VecArray,
    },
};

use crate::{
    compile::{
        CompileGraph, CompileTheory,
        analysis::{
            layering::{BoundaryFiberPoint, BoundaryRelation, NestedGraph, tensor_nested_graphs},
            partition::{OperationId, OperationRegion, RegionKind},
            wires::is_interleaved_control_operation,
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    lang::Obj,
    pass::inline::Inline,
    union_find::UnionFind,
};

#[derive(Debug, Clone)]
pub struct ControlRegionGraph {
    pub region_index: usize,
    pub region_operations: Vec<OperationId>,
    pub graph: Graph,
    pub morphism: ControlRegionMorphism,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlRegionMorphism {
    /// Result wire id -> parent region wire id, when the resolved wire is visible through the original interleaved control region.
    pub wires: Vec<Option<NodeId>>,
    /// Result operation id -> parent interleaved control operation id.
    pub operations: Vec<OperationId>,
}

#[derive(Debug, Clone)]
struct ResolvedGraph {
    graph: Graph,
    morphism: ControlRegionMorphism,
}

pub(super) fn process_control_regions(
    parent: &CompileGraph,
    regions: &[OperationRegion],
) -> Vec<ControlRegionGraph> {
    regions
        .iter()
        .enumerate()
        .filter(|(_, region)| matches!(region.kind, RegionKind::InterleavedControl))
        .map(|(region_index, region)| expand_control_region(parent, region_index, region))
        .collect()
}

// Expand one interleaved control region from a data graph into a native control graph, preserving a non-injective morphism back to the original region.
fn expand_control_region(
    parent: &CompileGraph,
    region_index: usize,
    region: &OperationRegion,
) -> ControlRegionGraph {
    let nested_graph = tensor_nested_graphs(
        region
            .operations
            .iter()
            .copied()
            .map(|operation_id| expand_interleaved_control_call(parent, operation_id))
            .collect::<Vec<_>>(),
    );
    let resolved = quotient_nested_graph(
        &parent.graph,
        nested_graph,
        Vec::new(),
        Vec::new(),
        "control region quotient",
    );

    ControlRegionGraph {
        region_index,
        region_operations: region.operations.clone(),
        graph: resolved.graph,
        morphism: resolved.morphism,
    }
}

// Expand a `control.foo` operation from a data graph. The native control child
// may have many boundary wires, but the interleaved call site can expose them
// through packed unary wires. Only child boundary projections are remapped to
// the call site; internal wires stay unmapped.
fn expand_interleaved_control_call(
    parent: &CompileGraph,
    operation_id: OperationId,
) -> NestedGraph {
    debug_assert!(is_interleaved_control_operation(
        &parent.graph,
        operation_id
    ));
    let operation = operation_name(&parent.graph, operation_id);
    let native_control_child =
        control_definition_for_operation(parent, operation).unwrap_or_else(|| {
            panic!(
                "interleaved control operation `{operation}` must resolve to a native control graph"
            )
        });

    let expanded_control_graph = inline_control_definitions(native_control_child);

    let expanded_source_wires = boundary_table(&expanded_control_graph.s);
    let expanded_target_wires = boundary_table(&expanded_control_graph.t);
    let call_inputs = operation_inputs(&parent.graph, operation_id).collect::<Vec<_>>();
    let call_outputs = operation_outputs(&parent.graph, operation_id).collect::<Vec<_>>();
    let boundary_relation = BoundaryRelation::from_boundaries(
        operation_id,
        (&expanded_source_wires, &call_inputs),
        (&expanded_target_wires, &call_outputs),
    );

    NestedGraph::under_parent_operation(operation_id, expanded_control_graph, boundary_relation)
}

// Expand a native control graph by inlining non-primitive control operations
// inside it. Mapping the expanded boundary to an interleaved call site is handled
// by `expand_interleaved_control_call`.
fn inline_control_definitions(graph: &CompileGraph) -> Graph {
    debug_assert!(matches!(graph.theory, CompileTheory::Control));
    let definitions = graph
        .children
        .iter()
        .filter(|child| matches!(child.graph.theory, CompileTheory::Control))
        .map(|child| {
            (
                child
                    .operation
                    .parse()
                    .expect("control child operation name must be valid"),
                LaxOpenHypergraph::from_strict(inline_control_definitions(&child.graph)),
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
        "too many control-definition inline iterations while expanding `{}`",
        graph.definition_name
    )
}

// Flatten a nested graph and quotient wires whose boundary-relation
// fiber points agree. The resulting graph keeps a public morphism to target
// wires by forgetting the fiber position and remembering only the data-theory
// wire.
fn quotient_nested_graph(
    target: &Graph,
    nested_graph: NestedGraph,
    source_boundary: Vec<NodeId>,
    target_boundary: Vec<NodeId>,
    context: &str,
) -> ResolvedGraph {
    nested_graph.validate_against_parent(target);

    let mut wires = Vec::<Obj>::new();
    let mut operations = Vec::new();
    let mut source_lengths = Vec::new();
    let mut target_lengths = Vec::new();
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut fiber_point_to_global_wire = HashMap::<BoundaryFiberPoint, usize>::new();
    let mut global_fiber_points = Vec::<Option<BoundaryFiberPoint>>::new();
    let mut duplicate_fiber_point_pairs = Vec::<(usize, usize)>::new();
    let mut operation_projection = Vec::new();

    for boundary_wire in source_boundary.iter().copied() {
        ensure_parent_boundary_wire(
            target,
            boundary_wire,
            &mut wires,
            &mut global_fiber_points,
            &mut fiber_point_to_global_wire,
        );
    }
    for boundary_wire in target_boundary.iter().copied() {
        ensure_parent_boundary_wire(
            target,
            boundary_wire,
            &mut wires,
            &mut global_fiber_points,
            &mut fiber_point_to_global_wire,
        );
    }

    let base = wires.len();
    let nested_fiber_points = nested_graph
        .boundary_relation
        .fiber_points_by_wire(nested_graph.graph.h.w.0.len());
    for (wire, fiber_point) in nested_graph
        .graph
        .h
        .w
        .0
        .0
        .iter()
        .cloned()
        .zip(nested_fiber_points.iter().copied())
    {
        let global = wires.len();
        wires.push(match fiber_point {
            Some(fiber_point) => target.h.w.0.0[fiber_point.parent_wire.0].clone(),
            None => wire,
        });
        global_fiber_points.push(fiber_point);
        if let Some(fiber_point) = fiber_point {
            if let Some(previous) = fiber_point_to_global_wire.get(&fiber_point).copied() {
                duplicate_fiber_point_pairs.push((previous, global));
            } else {
                fiber_point_to_global_wire.insert(fiber_point, global);
            }
        }
    }

    for operation_id in 0..nested_graph.graph.h.x.0.len() {
        operations.push(nested_graph.graph.h.x.0.0[operation_id].clone());
        operation_projection.push(nested_graph.parent_operations[operation_id]);
        let sources = operation_sources(&nested_graph.graph, operation_id)
            .into_iter()
            .map(|wire| base + wire.0)
            .collect::<Vec<_>>();
        let targets = operation_targets(&nested_graph.graph, operation_id)
            .into_iter()
            .map(|wire| base + wire.0)
            .collect::<Vec<_>>();
        source_lengths.push(sources.len());
        target_lengths.push(targets.len());
        source_values.extend(sources);
        target_values.extend(targets);
    }

    let mut uf = UnionFind::new(wires.len());
    for (left, right) in duplicate_fiber_point_pairs {
        uf.union(left, right);
    }

    let (class_by_wire, class_fiber_points, class_labels) =
        quotient_classes(&mut uf, &wires, &global_fiber_points, target);
    let source_values = source_values
        .into_iter()
        .map(|wire| class_by_wire[wire])
        .collect::<Vec<_>>();
    let target_values = target_values
        .into_iter()
        .map(|wire| class_by_wire[wire])
        .collect::<Vec<_>>();
    let source_boundary = source_boundary
        .into_iter()
        .map(|wire| {
            let fiber_point = BoundaryFiberPoint {
                parent_wire: wire,
                fiber_position: 0,
            };
            class_by_wire[fiber_point_to_global_wire[&fiber_point]]
        })
        .collect::<Vec<_>>();
    let target_boundary = target_boundary
        .into_iter()
        .map(|wire| {
            let fiber_point = BoundaryFiberPoint {
                parent_wire: wire,
                fiber_position: 0,
            };
            class_by_wire[fiber_point_to_global_wire[&fiber_point]]
        })
        .collect::<Vec<_>>();
    let wire_count = class_labels.len();

    let graph = OpenHypergraph {
        s: finite_function(source_boundary, wire_count),
        t: finite_function(target_boundary, wire_count),
        h: Hypergraph {
            s: indexed_coproduct(source_lengths, source_values, wire_count),
            t: indexed_coproduct(target_lengths, target_values, wire_count),
            w: SemifiniteFunction::new(VecArray(class_labels)),
            x: SemifiniteFunction::new(VecArray(operations)),
        },
    }
    .validate()
    .unwrap_or_else(|error| panic!("{context} produced invalid open hypergraph: {error:?}"));

    ResolvedGraph {
        graph,
        morphism: ControlRegionMorphism {
            wires: class_fiber_points
                .iter()
                .map(|fiber_point| fiber_point.map(|fiber_point| fiber_point.parent_wire))
                .collect(),
            operations: operation_projection,
        },
    }
}

fn ensure_parent_boundary_wire(
    target: &Graph,
    wire: NodeId,
    wires: &mut Vec<Obj>,
    global_fiber_points: &mut Vec<Option<BoundaryFiberPoint>>,
    fiber_point_to_global_wire: &mut HashMap<BoundaryFiberPoint, usize>,
) {
    let fiber_point = BoundaryFiberPoint {
        parent_wire: wire,
        fiber_position: 0,
    };
    if fiber_point_to_global_wire.contains_key(&fiber_point) {
        return;
    }
    let global = wires.len();
    wires.push(target.h.w.0.0[wire.0].clone());
    global_fiber_points.push(Some(fiber_point));
    fiber_point_to_global_wire.insert(fiber_point, global);
}

fn quotient_classes(
    uf: &mut UnionFind,
    wires: &[Obj],
    fiber_points: &[Option<BoundaryFiberPoint>],
    target: &Graph,
) -> (Vec<usize>, Vec<Option<BoundaryFiberPoint>>, Vec<Obj>) {
    let mut class_by_root = HashMap::<usize, usize>::new();
    let mut class_by_wire = vec![0; wires.len()];
    let mut class_fiber_points = Vec::<Option<BoundaryFiberPoint>>::new();
    let mut class_labels = Vec::<Obj>::new();

    for wire in 0..wires.len() {
        let root = uf.find(wire);
        let class = *class_by_root.entry(root).or_insert_with(|| {
            let fiber_point = fiber_points[wire];
            class_fiber_points.push(fiber_point);
            class_labels.push(match fiber_point {
                Some(fiber_point) => target.h.w.0.0[fiber_point.parent_wire.0].clone(),
                None => wires[wire].clone(),
            });
            class_fiber_points.len() - 1
        });
        if class_fiber_points[class].is_none()
            && let Some(fiber_point) = fiber_points[wire]
        {
            class_fiber_points[class] = Some(fiber_point);
            class_labels[class] = target.h.w.0.0[fiber_point.parent_wire.0].clone();
        }
        class_by_wire[wire] = class;
    }

    (class_by_wire, class_fiber_points, class_labels)
}

// Look up the child graph that gives the native control definition for an operation. Operations without such a child are treated as primitives.
fn control_definition_for_operation<'a>(
    graph: &'a CompileGraph,
    operation: &str,
) -> Option<&'a CompileGraph> {
    graph
        .children
        .iter()
        .find(|child| child.operation == operation)
        .map(|child| &child.graph)
        .filter(|child| matches!(child.theory, CompileTheory::Control))
}

fn boundary_table(boundary: &FiniteFunction) -> Vec<NodeId> {
    boundary.table.0.iter().copied().map(NodeId).collect()
}

fn operation_sources(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_inputs(graph, operation_id).collect()
}

fn operation_targets(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_outputs(graph, operation_id).collect()
}

fn indexed_coproduct(
    segment_lengths: Vec<usize>,
    values: Vec<usize>,
    target: usize,
) -> IndexedCoproduct<FiniteFunction> {
    let total = segment_lengths.iter().sum::<usize>();
    debug_assert_eq!(total, values.len());
    let sources = FiniteFunction::new(VecArray(segment_lengths), total + 1)
        .expect("segment lengths must form a valid indexed coproduct");
    let values = finite_function(values, target);
    IndexedCoproduct::new(sources, values).expect("incidence must be valid")
}

fn finite_function(table: Vec<usize>, target: usize) -> FiniteFunction {
    FiniteFunction::new(VecArray(table), target).expect("finite function table must be valid")
}
