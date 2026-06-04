use std::collections::HashMap;

use open_hypergraphs::{
    lax::NodeId,
    strict::vec::{
        FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph, SemifiniteFunction, VecArray,
    },
};

use crate::{
    compile::{
        CompileGraph, CompileTheory,
        analysis::{
            partition::{OperationId, OperationRegion, RegionKind},
            wires::is_interleaved_control_operation,
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    lang::Obj,
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
    projection: Vec<Option<ProjectionKey>>,
}

#[derive(Debug, Clone)]
struct TensorPiece {
    graph: Graph,
    wire_projection: Vec<Option<ProjectionKey>>,
    operation_projection: Vec<OperationId>,
}

// Internal quotient key. The public morphism only remembers the original
// region wire, but quotienting also needs branch position so packed branch
// alternatives are not identified too early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ProjectionKey {
    wire: NodeId,
    branch: Option<usize>,
}

impl ProjectionKey {
    fn plain(wire: NodeId) -> Self {
        Self { wire, branch: None }
    }

    fn branch(wire: NodeId, branch: usize) -> Self {
        Self {
            wire,
            branch: Some(branch),
        }
    }
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
    let pieces = region
        .operations
        .iter()
        .copied()
        .map(|operation_id| expand_interleaved_control_call(parent, operation_id))
        .collect::<Vec<_>>();
    let resolved = quotient_resolved_pieces(
        &parent.graph,
        pieces,
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

// Expand a `control.foo` operation from a data graph. If it has a child control definition, recursively expand that definition and remap its boundary projection to the unary interleaved call-site wire.
fn expand_interleaved_control_call(
    parent: &CompileGraph,
    operation_id: OperationId,
) -> TensorPiece {
    debug_assert!(is_interleaved_control_operation(
        &parent.graph,
        operation_id
    ));
    let operation = operation_name(&parent.graph, operation_id);
    let child = control_definition_for_operation(parent, operation).unwrap_or_else(|| {
        panic!("interleaved control operation `{operation}` must resolve to a native control graph")
    });

    let resolved = expand_control_definition(child);
    let wire_projection = remap_child_projection(&parent.graph, operation_id, child, &resolved);

    TensorPiece {
        graph: resolved.graph,
        wire_projection,
        operation_projection: vec![operation_id; resolved.morphism.operations.len()],
    }
}

// Recursively expand a native control definition by expanding each operation independently, tensoring those pieces, then quotienting according to the original control graph's wiring and open boundary.
fn expand_control_definition(graph: &CompileGraph) -> ResolvedGraph {
    debug_assert!(matches!(graph.theory, CompileTheory::Control));
    let pieces = (0..graph.graph.h.x.0.len())
        .map(|operation_id| expand_control_operation(graph, operation_id))
        .collect::<Vec<_>>();
    quotient_resolved_pieces(
        &graph.graph,
        pieces,
        boundary_table(&graph.graph.s),
        boundary_table(&graph.graph.t),
        "control graph quotient",
    )
}

// Expand one operation inside a native control definition. Child control definitions are inlined recursively; primitives remain as single-edge pieces.
fn expand_control_operation(graph: &CompileGraph, operation_id: OperationId) -> TensorPiece {
    let operation = operation_name(&graph.graph, operation_id);
    if let Some(child) = control_definition_for_operation(graph, operation) {
        let resolved = expand_control_definition(child);
        let wire_projection = remap_child_projection(&graph.graph, operation_id, child, &resolved);
        return TensorPiece {
            graph: resolved.graph,
            wire_projection,
            operation_projection: vec![operation_id; resolved.morphism.operations.len()],
        };
    }

    let graph = &graph.graph;
    let inputs = operation_inputs(graph, operation_id).collect::<Vec<_>>();
    let outputs = operation_outputs(graph, operation_id).collect::<Vec<_>>();
    let mut local_wire_by_host_wire = HashMap::<NodeId, usize>::new();
    let mut wires = Vec::<Obj>::new();
    let mut wire_projection = Vec::<Option<ProjectionKey>>::new();
    for wire in inputs.iter().chain(&outputs).copied() {
        local_wire_by_host_wire.entry(wire).or_insert_with(|| {
            let local = wires.len();
            wires.push(graph.h.w.0.0[wire.0].clone());
            wire_projection.push(Some(ProjectionKey::plain(wire)));
            local
        });
    }

    let source_values = inputs
        .iter()
        .map(|wire| local_wire_by_host_wire[wire])
        .collect::<Vec<_>>();
    let target_values = outputs
        .iter()
        .map(|wire| local_wire_by_host_wire[wire])
        .collect::<Vec<_>>();
    let wire_count = wires.len();
    let graph = OpenHypergraph {
        s: finite_function(source_values.clone(), wire_count),
        t: finite_function(target_values.clone(), wire_count),
        h: Hypergraph {
            s: indexed_coproduct(vec![source_values.len()], source_values, wire_count),
            t: indexed_coproduct(vec![target_values.len()], target_values, wire_count),
            w: SemifiniteFunction::new(VecArray(wires)),
            x: SemifiniteFunction::new(VecArray(vec![
                operation_name(graph, operation_id)
                    .parse()
                    .expect("resolved control operation name must be valid"),
            ])),
        },
    };

    TensorPiece {
        graph,
        wire_projection,
        operation_projection: vec![operation_id],
    }
}

// Tensor expanded pieces and quotient wires whose projection keys agree. The resulting graph keeps a public morphism to target wires, while branch-sensitive projection keys control which packed branch alternatives are identified.
fn quotient_resolved_pieces(
    target: &Graph,
    pieces: Vec<TensorPiece>,
    source_boundary: Vec<NodeId>,
    target_boundary: Vec<NodeId>,
    context: &str,
) -> ResolvedGraph {
    let mut wires = Vec::<Obj>::new();
    let mut operations = Vec::new();
    let mut source_lengths = Vec::new();
    let mut target_lengths = Vec::new();
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut projected_wire_to_global = HashMap::<ProjectionKey, usize>::new();
    let mut global_projection = Vec::<Option<ProjectionKey>>::new();
    let mut duplicate_projection_pairs = Vec::<(usize, usize)>::new();
    let mut operation_projection = Vec::new();

    for boundary_wire in source_boundary.iter().chain(&target_boundary).copied() {
        ensure_projected_wire(
            target,
            boundary_wire,
            &mut wires,
            &mut global_projection,
            &mut projected_wire_to_global,
        );
    }

    for piece in pieces {
        let base = wires.len();
        for (wire, projection) in piece
            .graph
            .h
            .w
            .0
            .0
            .iter()
            .cloned()
            .zip(piece.wire_projection.iter().copied())
        {
            let global = wires.len();
            wires.push(match projection {
                Some(projected) => target.h.w.0.0[projected.wire.0].clone(),
                None => wire,
            });
            global_projection.push(projection);
            if let Some(projected) = projection {
                if let Some(previous) = projected_wire_to_global.get(&projected).copied() {
                    duplicate_projection_pairs.push((previous, global));
                } else {
                    projected_wire_to_global.insert(projected, global);
                }
            }
        }

        for operation_id in 0..piece.graph.h.x.0.len() {
            operations.push(piece.graph.h.x.0.0[operation_id].clone());
            let sources = operation_sources(&piece.graph, operation_id)
                .into_iter()
                .map(|wire| base + wire.0)
                .collect::<Vec<_>>();
            let targets = operation_targets(&piece.graph, operation_id)
                .into_iter()
                .map(|wire| base + wire.0)
                .collect::<Vec<_>>();
            source_lengths.push(sources.len());
            target_lengths.push(targets.len());
            source_values.extend(sources);
            target_values.extend(targets);
        }
        operation_projection.extend(piece.operation_projection);
    }

    let mut uf = UnionFind::new(wires.len());
    for (left, right) in duplicate_projection_pairs {
        uf.union(left, right);
    }

    let (class_by_wire, class_projection, class_labels) =
        quotient_classes(&mut uf, &wires, &global_projection, target);
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
        .map(|wire| class_by_wire[projected_wire_to_global[&ProjectionKey::plain(wire)]])
        .collect::<Vec<_>>();
    let target_boundary = target_boundary
        .into_iter()
        .map(|wire| class_by_wire[projected_wire_to_global[&ProjectionKey::plain(wire)]])
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
        projection: class_projection.clone(),
        morphism: ControlRegionMorphism {
            wires: class_projection
                .iter()
                .map(|projection| projection.map(|projection| projection.wire))
                .collect(),
            operations: operation_projection,
        },
    }
}

fn ensure_projected_wire(
    target: &Graph,
    wire: NodeId,
    wires: &mut Vec<Obj>,
    global_projection: &mut Vec<Option<ProjectionKey>>,
    projected_wire_to_global: &mut HashMap<ProjectionKey, usize>,
) {
    let projection = ProjectionKey::plain(wire);
    if projected_wire_to_global.contains_key(&projection) {
        return;
    }
    let global = wires.len();
    wires.push(target.h.w.0.0[wire.0].clone());
    global_projection.push(Some(projection));
    projected_wire_to_global.insert(projection, global);
}

fn quotient_classes(
    uf: &mut UnionFind,
    wires: &[Obj],
    projections: &[Option<ProjectionKey>],
    target: &Graph,
) -> (Vec<usize>, Vec<Option<ProjectionKey>>, Vec<Obj>) {
    let mut class_by_root = HashMap::<usize, usize>::new();
    let mut class_by_wire = vec![0; wires.len()];
    let mut class_projection = Vec::<Option<ProjectionKey>>::new();
    let mut class_labels = Vec::<Obj>::new();

    for wire in 0..wires.len() {
        let root = uf.find(wire);
        let class = *class_by_root.entry(root).or_insert_with(|| {
            let projection = projections[wire];
            class_projection.push(projection);
            class_labels.push(match projection {
                Some(projected) => target.h.w.0.0[projected.wire.0].clone(),
                None => wires[wire].clone(),
            });
            class_projection.len() - 1
        });
        if class_projection[class].is_none()
            && let Some(projected) = projections[wire]
        {
            class_projection[class] = Some(projected);
            class_labels[class] = target.h.w.0.0[projected.wire.0].clone();
        }
        class_by_wire[wire] = class;
    }

    (class_by_wire, class_projection, class_labels)
}

// Translate a recursively expanded child definition's projection into the enclosing call graph. This is where multi-wire native control boundaries map through unary interleaved `control.foo` call boundaries.
fn remap_child_projection(
    call_graph: &Graph,
    call_operation_id: OperationId,
    child: &CompileGraph,
    resolved: &ResolvedGraph,
) -> Vec<Option<ProjectionKey>> {
    let source_wires = boundary_table(&child.graph.s);
    let target_wires = boundary_table(&child.graph.t);
    let call_inputs = operation_inputs(call_graph, call_operation_id).collect::<Vec<_>>();
    let call_outputs = operation_outputs(call_graph, call_operation_id).collect::<Vec<_>>();

    resolved
        .projection
        .iter()
        .map(|projection| {
            let projection = (*projection)?;
            boundary_projection(projection.wire, &source_wires, &call_inputs)
                .or_else(|| boundary_projection(projection.wire, &target_wires, &call_outputs))
                .or(Some(projection))
        })
        .collect()
}

fn boundary_projection(
    child_wire: NodeId,
    child_boundary: &[NodeId],
    call_boundary: &[NodeId],
) -> Option<ProjectionKey> {
    // A unary call boundary is a packed handle for all native child boundary alternatives. Keep the original call wire for the public morphism, but include the child boundary index in the quotient key.
    if call_boundary.len() == 1 && child_boundary.len() > 1 {
        return child_boundary
            .iter()
            .position(|wire| *wire == child_wire)
            .map(|branch| ProjectionKey::branch(call_boundary[0], branch));
    }

    child_boundary
        .iter()
        .position(|wire| *wire == child_wire)
        .and_then(|index| call_boundary.get(index).copied())
        .map(ProjectionKey::plain)
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
