use std::collections::HashMap;

use open_hypergraphs::{
    lax::NodeId,
    strict::vec::{
        FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph, SemifiniteFunction, VecArray,
    },
};

use crate::{
    compile::{
        analysis::partition::OperationId,
        graph_ops::{Graph, operation_inputs, operation_outputs},
    },
    lang::Obj,
    union_find::UnionFind,
};

// A span from a child graph boundary to its parent graph boundary. The vector
// index is the apex element; `child_wires` and `parent_wires` are the two legs.
#[derive(Debug, Clone)]
pub struct BoundaryRelation {
    apexes: Vec<BoundaryApex>,
    pub(super) child_wires: Vec<NodeId>,
    pub(super) parent_wires: Vec<NodeId>,
}

impl BoundaryRelation {
    pub(super) fn from_boundaries(
        parent_operation: OperationId,
        source: (&[NodeId], &[NodeId]),
        target: (&[NodeId], &[NodeId]),
    ) -> Self {
        let links = boundary_links(parent_operation, BoundarySide::Source, source.0, source.1)
            .chain(boundary_links(
                parent_operation,
                BoundarySide::Target,
                target.0,
                target.1,
            ))
            .fold(
                Vec::<(BoundaryApex, NodeId, NodeId)>::new(),
                |mut links, link| {
                    if !links.iter().any(|(_, child_wire, _)| *child_wire == link.1) {
                        links.push(link);
                    }
                    links
                },
            );
        Self::from_links(links)
    }

    fn empty() -> Self {
        Self {
            apexes: Vec::new(),
            child_wires: Vec::new(),
            parent_wires: Vec::new(),
        }
    }

    fn from_links(links: Vec<(BoundaryApex, NodeId, NodeId)>) -> Self {
        let mut apexes = Vec::new();
        let mut child_wires = Vec::new();
        let mut parent_wires = Vec::new();
        for (apex, child_wire, parent_wire) in links {
            apexes.push(apex);
            child_wires.push(child_wire);
            parent_wires.push(parent_wire);
        }
        Self {
            apexes,
            child_wires,
            parent_wires,
        }
    }

    fn shifted_child_wires(
        &self,
        offset: usize,
    ) -> impl Iterator<Item = (BoundaryApex, NodeId, NodeId)> + '_ {
        self.apexes
            .iter()
            .copied()
            .zip(self.child_wires.iter().copied())
            .zip(self.parent_wires.iter().copied())
            .map(move |((apex, child_wire), parent_wire)| {
                (apex, NodeId(child_wire.0 + offset), parent_wire)
            })
    }

    pub(super) fn fiber_points_by_wire(
        &self,
        wire_count: usize,
    ) -> Vec<Option<BoundaryFiberPoint>> {
        debug_assert_eq!(self.apexes.len(), self.child_wires.len());
        debug_assert_eq!(self.child_wires.len(), self.parent_wires.len());
        let mut fiber_positions = HashMap::<BoundaryFiber, usize>::new();
        let mut fiber_points = vec![None; wire_count];
        for ((apex, child_wire), parent_wire) in self
            .apexes
            .iter()
            .copied()
            .zip(self.child_wires.iter().copied())
            .zip(self.parent_wires.iter().copied())
        {
            let position = fiber_positions
                .entry(BoundaryFiber {
                    parent_operation: apex.parent_operation,
                    parent_wire,
                    side: apex.side,
                })
                .or_default();
            fiber_points[child_wire.0] = Some(BoundaryFiberPoint {
                parent_wire,
                fiber_position: *position,
            });
            *position += 1;
        }
        fiber_points
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundaryApex {
    parent_operation: OperationId,
    side: BoundarySide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum BoundarySide {
    Source,
    Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundaryFiber {
    parent_operation: OperationId,
    parent_wire: NodeId,
    side: BoundarySide,
}

#[derive(Debug, Clone)]
pub struct NestedGraph {
    pub graph: Graph,
    pub parent_operations: Vec<OperationId>,
    pub boundary_relation: BoundaryRelation,
}

impl NestedGraph {
    pub(super) fn under_parent_operation(
        parent_operation: OperationId,
        graph: Graph,
        boundary_relation: BoundaryRelation,
    ) -> Self {
        let operation_count = graph.h.x.0.len();
        Self {
            graph,
            parent_operations: vec![parent_operation; operation_count],
            boundary_relation,
        }
    }

    pub(super) fn validate_against_parent(&self, parent: &Graph) {
        assert_eq!(
            self.parent_operations.len(),
            self.graph.h.x.0.len(),
            "nested graph must map every child operation to a parent operation"
        );
        assert_eq!(
            self.boundary_relation.child_wires.len(),
            self.boundary_relation.parent_wires.len(),
            "boundary relation legs must have the same apex size"
        );
        assert_eq!(
            self.boundary_relation.apexes.len(),
            self.boundary_relation.child_wires.len(),
            "boundary relation apexes and legs must have the same size"
        );
        for parent_operation in &self.parent_operations {
            assert!(
                *parent_operation < parent.h.x.0.len(),
                "nested graph maps to missing parent operation {parent_operation}"
            );
        }
        for child_wire in &self.boundary_relation.child_wires {
            assert!(
                child_wire.0 < self.graph.h.w.0.len(),
                "boundary relation references missing child wire {:?}",
                child_wire
            );
        }
        for parent_wire in &self.boundary_relation.parent_wires {
            assert!(
                parent_wire.0 < parent.h.w.0.len(),
                "boundary relation references missing parent wire {:?}",
                parent_wire
            );
        }

        for ((apex, child_wire), parent_wire) in self
            .boundary_relation
            .apexes
            .iter()
            .zip(&self.boundary_relation.child_wires)
            .zip(&self.boundary_relation.parent_wires)
        {
            let parent_boundary = operation_side_boundary(parent, apex.parent_operation, apex.side);
            assert!(
                parent_boundary.contains(parent_wire),
                "boundary relation maps to parent wire {:?}, which is not on {:?} boundary of parent operation {}",
                parent_wire,
                apex.side,
                apex.parent_operation
            );
            let compatible_child_operation = (0..self.graph.h.x.0.len()).any(|child_operation| {
                if self.parent_operations[child_operation] != apex.parent_operation {
                    return false;
                }
                let child_boundary =
                    operation_side_boundary(&self.graph, child_operation, apex.side);
                if child_boundary.contains(child_wire) {
                    return true;
                }
                false
            });
            assert!(
                compatible_child_operation,
                "boundary relation child wire {:?} has no child operation on {:?} boundary mapped to parent operation {}",
                child_wire, apex.side, apex.parent_operation
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BoundaryFiberPoint {
    pub(super) parent_wire: NodeId,
    pub(super) fiber_position: usize,
}

pub(super) fn coproduct_over_parent(nested_graphs: Vec<NestedGraph>) -> NestedGraph {
    let mut wires = Vec::<Obj>::new();
    let mut operations = Vec::new();
    let mut source_lengths = Vec::new();
    let mut target_lengths = Vec::new();
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut parent_operations = Vec::new();
    let mut boundary_links = Vec::<(BoundaryApex, NodeId, NodeId)>::new();

    for nested_graph in nested_graphs {
        let wire_base = wires.len();
        wires.extend(nested_graph.graph.h.w.0.0.iter().cloned());
        boundary_links.extend(
            nested_graph
                .boundary_relation
                .shifted_child_wires(wire_base),
        );

        for operation_id in 0..nested_graph.graph.h.x.0.len() {
            operations.push(nested_graph.graph.h.x.0.0[operation_id].clone());
            let sources = operation_sources(&nested_graph.graph, operation_id)
                .into_iter()
                .map(|wire| wire.0 + wire_base)
                .collect::<Vec<_>>();
            let targets = operation_targets(&nested_graph.graph, operation_id)
                .into_iter()
                .map(|wire| wire.0 + wire_base)
                .collect::<Vec<_>>();
            source_lengths.push(sources.len());
            target_lengths.push(targets.len());
            source_values.extend(sources);
            target_values.extend(targets);
        }
        parent_operations.extend(nested_graph.parent_operations);
    }

    let wire_count = wires.len();
    let graph = OpenHypergraph {
        s: finite_function(Vec::new(), wire_count),
        t: finite_function(Vec::new(), wire_count),
        h: Hypergraph {
            s: indexed_coproduct(source_lengths, source_values, wire_count),
            t: indexed_coproduct(target_lengths, target_values, wire_count),
            w: SemifiniteFunction::new(VecArray(wires)),
            x: SemifiniteFunction::new(VecArray(operations)),
        },
    }
    .validate()
    .expect("coproduct of nested graphs must be valid");

    NestedGraph {
        graph,
        parent_operations,
        boundary_relation: if boundary_links.is_empty() {
            BoundaryRelation::empty()
        } else {
            BoundaryRelation::from_links(boundary_links)
        },
    }
}

pub(super) fn quotient_over_parent(parent: &Graph, nested_graph: NestedGraph) -> NestedGraph {
    nested_graph.validate_against_parent(parent);

    let mut wires = Vec::<Obj>::new();
    let mut operations = Vec::new();
    let mut source_lengths = Vec::new();
    let mut target_lengths = Vec::new();
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut fiber_point_to_global_wire = HashMap::<BoundaryFiberPoint, usize>::new();
    let mut global_fiber_points = Vec::<Option<BoundaryFiberPoint>>::new();
    let mut duplicate_fiber_point_pairs = Vec::<(usize, usize)>::new();

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
            Some(fiber_point) => parent.h.w.0.0[fiber_point.parent_wire.0].clone(),
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
        let sources = operation_sources(&nested_graph.graph, operation_id)
            .into_iter()
            .map(|wire| wire.0)
            .collect::<Vec<_>>();
        let targets = operation_targets(&nested_graph.graph, operation_id)
            .into_iter()
            .map(|wire| wire.0)
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

    let (class_by_wire, class_labels) =
        quotient_classes(&mut uf, &wires, &global_fiber_points, parent);
    let source_values = source_values
        .into_iter()
        .map(|wire| class_by_wire[wire])
        .collect::<Vec<_>>();
    let target_values = target_values
        .into_iter()
        .map(|wire| class_by_wire[wire])
        .collect::<Vec<_>>();
    let wire_count = class_labels.len();

    let graph = OpenHypergraph {
        s: finite_function(Vec::new(), wire_count),
        t: finite_function(Vec::new(), wire_count),
        h: Hypergraph {
            s: indexed_coproduct(source_lengths, source_values, wire_count),
            t: indexed_coproduct(target_lengths, target_values, wire_count),
            w: SemifiniteFunction::new(VecArray(class_labels)),
            x: SemifiniteFunction::new(VecArray(operations)),
        },
    }
    .validate()
    .expect("quotient of nested graph must be valid");

    let boundary_relation =
        quotient_boundary_relation(&nested_graph.boundary_relation, &class_by_wire);
    let result = NestedGraph {
        graph,
        parent_operations: nested_graph.parent_operations,
        boundary_relation,
    };
    result.validate_against_parent(parent);
    result
}

fn quotient_boundary_relation(
    relation: &BoundaryRelation,
    class_by_wire: &[usize],
) -> BoundaryRelation {
    let mut links = Vec::<(BoundaryApex, NodeId, NodeId)>::new();
    for ((apex, child_wire), parent_wire) in relation
        .apexes
        .iter()
        .copied()
        .zip(relation.child_wires.iter().copied())
        .zip(relation.parent_wires.iter().copied())
    {
        let child_wire = NodeId(class_by_wire[child_wire.0]);
        if !links
            .iter()
            .any(|link| *link == (apex, child_wire, parent_wire))
        {
            links.push((apex, child_wire, parent_wire));
        }
    }
    BoundaryRelation::from_links(links)
}

fn quotient_classes(
    uf: &mut UnionFind,
    wires: &[Obj],
    fiber_points: &[Option<BoundaryFiberPoint>],
    parent: &Graph,
) -> (Vec<usize>, Vec<Obj>) {
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
                Some(fiber_point) => parent.h.w.0.0[fiber_point.parent_wire.0].clone(),
                None => wires[wire].clone(),
            });
            class_fiber_points.len() - 1
        });
        if class_fiber_points[class].is_none()
            && let Some(fiber_point) = fiber_points[wire]
        {
            class_fiber_points[class] = Some(fiber_point);
            class_labels[class] = parent.h.w.0.0[fiber_point.parent_wire.0].clone();
        }
        class_by_wire[wire] = class;
    }

    (class_by_wire, class_labels)
}

fn boundary_links<'a>(
    parent_operation: OperationId,
    side: BoundarySide,
    child_boundary: &'a [NodeId],
    parent_boundary: &'a [NodeId],
) -> impl Iterator<Item = (BoundaryApex, NodeId, NodeId)> + 'a {
    child_boundary
        .iter()
        .copied()
        .enumerate()
        .filter_map(move |(index, child_wire)| {
            related_parent_wire(index, parent_boundary).map(|parent_wire| {
                (
                    BoundaryApex {
                        parent_operation,
                        side,
                    },
                    child_wire,
                    parent_wire,
                )
            })
        })
}

fn related_parent_wire(boundary_index: usize, parent_boundary: &[NodeId]) -> Option<NodeId> {
    if parent_boundary.len() == 1 {
        parent_boundary.first().copied()
    } else {
        parent_boundary.get(boundary_index).copied()
    }
}

fn operation_sources(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_inputs(graph, operation_id).collect()
}

fn operation_targets(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_outputs(graph, operation_id).collect()
}

fn operation_side_boundary(
    graph: &Graph,
    operation_id: OperationId,
    side: BoundarySide,
) -> Vec<NodeId> {
    match side {
        BoundarySide::Source => operation_inputs(graph, operation_id).collect(),
        BoundarySide::Target => operation_outputs(graph, operation_id).collect(),
    }
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
