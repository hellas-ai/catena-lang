use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::strict::vec::{
    FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph, SemifiniteFunction, VecArray,
};

use crate::{
    compile::{
        cfg::{
            layering::{BoundaryFiberPoint, BoundarySide, Layer, NestingMorphism, Region},
            partition::RegionKind,
        },
        graph_ops::{Graph, operation_inputs, operation_outputs},
    },
    lang::Obj,
    union_find::UnionFind,
};

pub(super) struct RegionGraph {
    pub(super) graph: Graph,
    pub(super) regions: Vec<RegionGraphRegion>,
}

pub(super) struct RegionGraphRegion {
    pub(super) path: Vec<usize>,
    pub(super) kind: RegionKind,
    pub(super) graph: Graph,
    pub(super) region: Region,
    pub(super) inputs: Vec<usize>,
    pub(super) outputs: Vec<usize>,
}

pub(super) fn region_graph(layer: &Layer) -> Graph {
    lower_layer_to_region_graph(layer).graph
}

pub(super) fn lower_layer_to_region_graph(layer: &Layer) -> RegionGraph {
    let mut builder = RegionGraphBuilder::default();
    builder.add_layer(layer);
    builder.finish()
}

pub(super) fn region_graph_trace(layer: &Layer) -> Vec<u8> {
    // Debug artifact for auditing the construction on examples. It prints both
    // the recursive layer tree and the final quotient graph incidence, e.g.:
    //
    //   region.1.0.control: (w1) -> (w2, w3)
    //   region.1.1.0.data: (w2) -> (w4)
    //   region.1.2.0.data: (w3) -> (w5)
    //   region.1.3.control: (w4, w5) -> (w6)
    let mut trace = String::new();
    append_layer_trace(&mut trace, layer, &[]);

    let graph = region_graph(layer);
    writeln!(&mut trace, "\nregion graph").expect("write to string cannot fail");
    for operation_id in 0..graph.h.x.0.len() {
        let sources = operation_inputs(&graph, operation_id)
            .map(|wire| format!("w{}", wire.0))
            .collect::<Vec<_>>()
            .join(", ");
        let targets = operation_outputs(&graph, operation_id)
            .map(|wire| format!("w{}", wire.0))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            &mut trace,
            "  {}: ({sources}) -> ({targets})",
            graph.h.x.0[operation_id]
        )
        .expect("write to string cannot fail");
    }

    trace.into_bytes()
}

#[derive(Default)]
struct RegionGraphBuilder {
    next_layer: usize,
    wire_class_by_layer_wire: HashMap<(usize, usize), usize>,
    wire_class_by_boundary_fiber: HashMap<(usize, usize, usize), usize>,
    wire_labels: Vec<Obj>,
    equations: Vec<(usize, usize)>,
    operations: Vec<RegionOperation>,
    regions: Vec<RegionGraphRegion>,
}

impl RegionGraphBuilder {
    fn add_layer(&mut self, layer: &Layer) {
        self.visit_layer(layer, None, Vec::new());
    }

    // Build a graph whose operations are the leaf regions of the layer tree.
    //
    // Example layer shape:
    //
    //   root
    //     region 0: Data                  -> operation region.0.data
    //     region 1: InterleavedControl    -> descend
    //       region 0: Control             -> operation region.1.0.control
    //       region 1: InterleavedData     -> descend
    //         region 0: Data              -> operation region.1.1.0.data
    //     region 2: Data                  -> operation region.2.data
    //
    // Interleaved regions are not operations here; they only explain where a
    // child layer is attached.
    fn visit_layer(&mut self, layer: &Layer, parent_layer: Option<usize>, path: Vec<usize>) {
        let layer_id = self.alloc_layer(layer);
        self.connect_nested_layer_boundary(layer, layer_id, parent_layer);

        let data_context = DataRegionInterfaceContext::new(
            &layer.graph,
            &layer.regions,
            layer.morphism_to_parent.as_ref(),
        );
        for region in &layer.regions {
            self.visit_region(layer_id, &layer.graph, &data_context, region, &path);
        }
    }

    fn visit_region(
        &mut self,
        layer_id: usize,
        graph: &Graph,
        data_context: &DataRegionInterfaceContext,
        region: &Region,
        parent_path: &[usize],
    ) {
        let mut path = parent_path.to_vec();
        path.push(region.index);

        match (&region.kind, &region.expansion) {
            (_, Some(expansion)) => self.visit_layer(expansion, Some(layer_id), path),
            (RegionKind::Data, None) => {
                let interface = data_context.data_region_interface(graph, region);
                self.add_region_operation(layer_id, graph, region, path, interface);
            }
            (RegionKind::Control, None) => {
                let interface = RegionInterface::control_region(graph, region);
                self.add_region_operation(layer_id, graph, region, path, interface);
            }
            (RegionKind::InterleavedControl | RegionKind::InterleavedData, None) => {
                panic!(
                    "interleaved regions must be expanded before becoming region graph operations"
                )
            }
        }
    }

    // Each layer graph has its own wire namespace. We allocate one provisional
    // wire class for each layer wire and later quotient these classes.
    //
    //   layer 3 wire 7  -> class c42
    //   layer 4 wire 7  -> class c99
    //
    // They are different until a boundary morphism or region-interface rule
    // explicitly equates them.
    fn alloc_layer(&mut self, layer: &Layer) -> usize {
        let layer_id = self.next_layer;
        self.next_layer += 1;

        for (wire, label) in layer.graph.h.w.0.0.iter().cloned().enumerate() {
            let class = self.wire_labels.len();
            self.wire_labels.push(label);
            self.wire_class_by_layer_wire
                .insert((layer_id, wire), class);
        }

        layer_id
    }

    // Connect a child layer back to its parent through the nesting morphism.
    //
    // The simple case is one child boundary wire for one parent boundary wire:
    //
    //   child w8  ~ parent w11
    //
    // Branching/merging needs more care. A packed parent wire may represent
    // several boundary fibers:
    //
    //   child w2 ~ parent w17, fiber 0
    //   child w3 ~ parent w17, fiber 1
    //
    // These must NOT be quotiented together. Otherwise a 1 -> 2 branch becomes
    // a 1 -> 1 self-connection:
    //
    //   wrong:  branch: w1 -> (w2, w2)
    //   right:  branch: w1 -> (w2, w3)
    //
    // So when a parent wire has multiple fibers we allocate one synthetic
    // parent-side class per fiber position.
    fn connect_nested_layer_boundary(
        &mut self,
        layer: &Layer,
        layer_id: usize,
        parent_layer: Option<usize>,
    ) {
        let Some(parent_layer) = parent_layer else {
            return;
        };
        let Some(morphism) = &layer.morphism_to_parent else {
            panic!("nested layer must carry a morphism to its parent")
        };

        let fiber_points = morphism
            .boundary_relation
            .fiber_points_by_wire(layer.graph.h.w.0.len());
        let multi_fiber_parent_wires = multi_fiber_parent_wires(&fiber_points);

        for (child_wire, fiber_point) in fiber_points.into_iter().enumerate() {
            let Some(fiber_point) = fiber_point else {
                continue;
            };
            self.connect_boundary_fiber(
                layer_id,
                child_wire,
                parent_layer,
                fiber_point.parent_wire.0,
                fiber_point.fiber_position,
                multi_fiber_parent_wires.contains(&fiber_point.parent_wire.0),
            );
        }
    }

    // Add the quotient equation for one child boundary fiber.
    fn connect_boundary_fiber(
        &mut self,
        child_layer: usize,
        child_wire: usize,
        parent_layer: usize,
        parent_wire: usize,
        fiber_position: usize,
        parent_wire_has_multiple_fibers: bool,
    ) {
        let child_class = self.layer_wire_class(child_layer, child_wire);
        let parent_class = if parent_wire_has_multiple_fibers {
            self.boundary_fiber_class(parent_layer, parent_wire, fiber_position)
        } else {
            self.layer_wire_class(parent_layer, parent_wire)
        };
        self.equations.push((child_class, parent_class));
    }

    // Emit one leaf-region operation using the already-selected interface.
    // Interface equations collapse multi-wire data boundaries into the single
    // data-control token we expose at this level:
    //
    //   data candidate inputs:  w2, w4, w9
    //   exposed input:          w2
    //   equations:              w2 ~ w4, w2 ~ w9
    //
    // The operation itself is still 1 -> 1.
    fn add_region_operation(
        &mut self,
        layer_id: usize,
        graph: &Graph,
        region: &Region,
        path: Vec<usize>,
        interface: RegionInterface,
    ) {
        let place_boundary = RegionBoundary::new(graph, region);
        let region_inputs = interface_place_wires(&interface.inputs)
            .unwrap_or_else(|| place_boundary.inputs.clone());
        let region_outputs = interface_place_wires(&interface.outputs)
            .unwrap_or_else(|| place_boundary.outputs.clone());

        for (left, right) in interface.equations {
            self.equations.push((
                self.layer_wire_class(layer_id, left),
                self.layer_wire_class(layer_id, right),
            ));
        }

        let sources = interface
            .inputs
            .into_iter()
            .map(|wire| self.interface_wire_class(layer_id, wire))
            .collect();
        let targets = interface
            .outputs
            .into_iter()
            .map(|wire| self.interface_wire_class(layer_id, wire))
            .collect();

        self.operations.push(RegionOperation {
            operation: region_operation(region, &path),
            sources,
            targets,
        });
        self.regions.push(RegionGraphRegion {
            path,
            kind: region.kind,
            graph: graph.clone(),
            region: region.clone(),
            inputs: region_inputs,
            outputs: region_outputs,
        });
    }

    fn layer_wire_class(&self, layer_id: usize, wire: usize) -> usize {
        *self
            .wire_class_by_layer_wire
            .get(&(layer_id, wire))
            .unwrap_or_else(|| {
                panic!("missing region graph wire class for layer {layer_id} wire {wire}")
            })
    }

    fn interface_wire_class(&mut self, layer_id: usize, wire: InterfaceWire) -> usize {
        match wire {
            InterfaceWire::Layer(wire) => self.layer_wire_class(layer_id, wire),
            InterfaceWire::Synthetic => {
                let class = self.wire_labels.len();
                self.wire_labels.push(Tree::Empty);
                class
            }
        }
    }

    fn boundary_fiber_class(
        &mut self,
        layer_id: usize,
        parent_wire: usize,
        fiber_position: usize,
    ) -> usize {
        if let Some(class) =
            self.wire_class_by_boundary_fiber
                .get(&(layer_id, parent_wire, fiber_position))
        {
            return *class;
        }

        let class = self.wire_labels.len();
        self.wire_labels.push(Tree::Empty);
        self.wire_class_by_boundary_fiber
            .insert((layer_id, parent_wire, fiber_position), class);
        class
    }

    // Convert provisional classes into actual graph wires. Only classes used
    // by emitted region-operation ports become graph wires, so internal layer
    // wires that are irrelevant to region connectivity disappear here.
    fn finish(self) -> RegionGraph {
        let mut uf = UnionFind::new(self.wire_labels.len());
        for (left, right) in self.equations {
            uf.union(left, right);
        }

        let used_classes = self
            .operations
            .iter()
            .flat_map(|operation| operation.sources.iter().chain(&operation.targets))
            .copied()
            .collect::<Vec<_>>();
        let (wire_by_class, wires) = quotient_wires(&mut uf, self.wire_labels, &used_classes);
        let mut operations = Vec::new();
        let mut source_lengths = Vec::new();
        let mut target_lengths = Vec::new();
        let mut source_values = Vec::new();
        let mut target_values = Vec::new();

        for operation in self.operations {
            operations.push(operation.operation);
            source_lengths.push(operation.sources.len());
            target_lengths.push(operation.targets.len());
            source_values.extend(
                operation
                    .sources
                    .into_iter()
                    .map(|class| wire_by_class[class]),
            );
            target_values.extend(
                operation
                    .targets
                    .into_iter()
                    .map(|class| wire_by_class[class]),
            );
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
        .expect("region graph must be valid");

        RegionGraph {
            graph,
            regions: self.regions,
        }
    }
}

struct RegionOperation {
    operation: Operation,
    sources: Vec<usize>,
    targets: Vec<usize>,
}

#[derive(Debug, Clone)]
struct RegionInterface {
    inputs: Vec<InterfaceWire>,
    outputs: Vec<InterfaceWire>,
    equations: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
enum InterfaceWire {
    Layer(usize),
    Synthetic,
}

impl RegionInterface {
    // Native control regions keep their real control-flow shape.
    //
    //   sequential:  1 -> 1
    //   branch:      1 -> 2
    //   merge:       2 -> 1
    //
    // Anything else means the partitioning did not produce CFG-shaped control
    // leaves.
    fn control_region(graph: &Graph, region: &Region) -> Self {
        let boundary = RegionBoundary::new(graph, region);
        match (boundary.inputs.len(), boundary.outputs.len()) {
            (1, 1) | (1, 2) | (2, 1) => Self {
                inputs: boundary
                    .inputs
                    .into_iter()
                    .map(InterfaceWire::Layer)
                    .collect(),
                outputs: boundary
                    .outputs
                    .into_iter()
                    .map(InterfaceWire::Layer)
                    .collect(),
                equations: Vec::new(),
            },
            _ => panic!(
                "unsupported control region graph interface: {} inputs -> {} outputs",
                boundary.inputs.len(),
                boundary.outputs.len()
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct DataRegionInterfaceContext {
    graph_sources: Vec<usize>,
    graph_targets: Vec<usize>,
    control_inputs: Vec<usize>,
    control_outputs: Vec<usize>,
}

impl DataRegionInterfaceContext {
    fn new(
        graph: &Graph,
        regions: &[Region],
        morphism_to_parent: Option<&NestingMorphism>,
    ) -> Self {
        let control_operations = regions
            .iter()
            .filter(|region| matches!(region.kind, RegionKind::InterleavedControl))
            .flat_map(|region| region.operations.iter().copied())
            .collect::<Vec<_>>();

        Self {
            graph_sources: child_boundary_wires(morphism_to_parent, BoundarySide::Source),
            graph_targets: child_boundary_wires(morphism_to_parent, BoundarySide::Target),
            control_inputs: unique_wires(
                control_operations.iter().copied().flat_map(|operation_id| {
                    operation_inputs(graph, operation_id).map(|wire| wire.0)
                }),
            ),
            control_outputs: unique_wires(control_operations.iter().copied().flat_map(
                |operation_id| operation_outputs(graph, operation_id).map(|wire| wire.0),
            )),
        }
    }

    // Data regions are exposed as 1 -> 1 even if their underlying data block
    // touches several concrete wires.
    //
    // We first find the data wires that matter for region connectivity:
    //
    //   entry wires = external layer sources or wires coming from control
    //   exit wires  = external layer targets or wires going to control
    //
    // Then we pick one representative on each side and quotient the rest.
    //
    //   candidates:  inputs  [w0, w3]   outputs [w8, w9]
    //   operation:   w0 -> w8
    //   equations:   w0 ~ w3, w8 ~ w9
    //
    // If a side has no region-connectivity wire, we use a synthetic port:
    //
    //   root producer:  _ -> w15
    //   final sink:     w11 -> _
    fn data_region_interface(&self, graph: &Graph, region: &Region) -> RegionInterface {
        let boundary = self.data_region_boundary(graph, region);
        let input = representative_or_synthetic(&boundary.inputs);
        let output = representative_or_synthetic(&boundary.outputs);
        let equations = collapse_boundary_wires(&boundary.inputs)
            .into_iter()
            .chain(collapse_boundary_wires(&boundary.outputs))
            .collect();

        RegionInterface {
            inputs: vec![input],
            outputs: vec![output],
            equations,
        }
    }

    // Data entry/exit wires must be outside the data region itself. This avoids
    // choosing unpacked internal component wires as block boundaries:
    //
    //   w1 = unpack(w0)
    //   ...
    //   w8 = pack(...)
    //
    // Here w1 may be related to the parent boundary, but it is produced and
    // consumed inside the data region, so it is not the region exit. The exit is
    // the produced-not-consumed packed value w8.
    fn data_region_boundary(&self, graph: &Graph, region: &Region) -> RegionBoundary {
        let consumed = region_consumed_wires(graph, region);
        let produced = region_produced_wires(graph, region);

        RegionBoundary {
            inputs: consumed
                .iter()
                .copied()
                .filter(|wire| {
                    !produced.contains(wire)
                        && (self.graph_sources.contains(wire)
                            || self.control_outputs.contains(wire))
                })
                .collect(),
            outputs: produced
                .iter()
                .copied()
                .filter(|wire| {
                    !consumed.contains(wire)
                        && (self.graph_targets.contains(wire) || self.control_inputs.contains(wire))
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
struct RegionBoundary {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl RegionBoundary {
    fn new(graph: &Graph, region: &Region) -> Self {
        let consumed = region_consumed_wires(graph, region);
        let produced = region_produced_wires(graph, region);
        Self::from_consumed_produced(graph, consumed, produced)
    }

    fn from_consumed_produced(graph: &Graph, consumed: Vec<usize>, produced: Vec<usize>) -> Self {
        let graph_sources = graph.s.table.iter().copied().collect::<Vec<_>>();
        let graph_targets = graph.t.table.iter().copied().collect::<Vec<_>>();

        Self {
            inputs: consumed
                .iter()
                .copied()
                .filter(|wire| !produced.contains(wire) || graph_sources.contains(wire))
                .collect(),
            outputs: produced
                .iter()
                .copied()
                .filter(|wire| !consumed.contains(wire) || graph_targets.contains(wire))
                .collect(),
        }
    }
}

fn region_consumed_wires(graph: &Graph, region: &Region) -> Vec<usize> {
    unique_wires(
        region
            .operations
            .iter()
            .copied()
            .flat_map(|operation_id| operation_inputs(graph, operation_id).map(|wire| wire.0)),
    )
}

fn region_produced_wires(graph: &Graph, region: &Region) -> Vec<usize> {
    unique_wires(
        region
            .operations
            .iter()
            .copied()
            .flat_map(|operation_id| operation_outputs(graph, operation_id).map(|wire| wire.0)),
    )
}

fn unique_wires(wires: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut unique = Vec::new();
    for wire in wires {
        if !unique.contains(&wire) {
            unique.push(wire);
        }
    }
    unique
}

fn representative_or_synthetic(wires: &[usize]) -> InterfaceWire {
    wires
        .first()
        .copied()
        .map(InterfaceWire::Layer)
        .unwrap_or(InterfaceWire::Synthetic)
}

fn interface_place_wires(wires: &[InterfaceWire]) -> Option<Vec<usize>> {
    let mut place_wires = Vec::new();
    for wire in wires {
        match wire {
            InterfaceWire::Layer(wire) => place_wires.push(*wire),
            InterfaceWire::Synthetic => return None,
        }
    }
    Some(place_wires)
}

fn collapse_boundary_wires(wires: &[usize]) -> Vec<(usize, usize)> {
    let Some(first) = wires.first().copied() else {
        return Vec::new();
    };
    wires
        .iter()
        .copied()
        .skip(1)
        .map(|wire| (first, wire))
        .collect()
}

fn child_boundary_wires(
    morphism_to_parent: Option<&NestingMorphism>,
    side: BoundarySide,
) -> Vec<usize> {
    morphism_to_parent
        .map(|morphism| {
            morphism
                .boundary_relation
                .child_wires_on_side(side)
                .into_iter()
                .map(|wire| wire.0)
                .collect()
        })
        .unwrap_or_default()
}

fn multi_fiber_parent_wires(fiber_points: &[Option<BoundaryFiberPoint>]) -> HashSet<usize> {
    let mut fibers_by_parent_wire = HashMap::<usize, HashSet<usize>>::new();
    for fiber_point in fiber_points.iter().flatten() {
        fibers_by_parent_wire
            .entry(fiber_point.parent_wire.0)
            .or_default()
            .insert(fiber_point.fiber_position);
    }

    fibers_by_parent_wire
        .into_iter()
        .filter_map(|(parent_wire, fibers)| (fibers.len() > 1).then_some(parent_wire))
        .collect()
}

fn quotient_wires(
    uf: &mut UnionFind,
    labels: Vec<Obj>,
    used_classes: &[usize],
) -> (Vec<usize>, Vec<Obj>) {
    let mut wire_by_root = HashMap::<usize, usize>::new();
    let mut wire_by_class = vec![0; labels.len()];
    let mut wires = Vec::new();

    for class in used_classes {
        let root = uf.find(*class);
        let wire = *wire_by_root.entry(root).or_insert_with(|| {
            let wire = wires.len();
            wires.push(Tree::Empty);
            wire
        });
        wire_by_class[*class] = wire;
    }

    for class in used_classes {
        let root = uf.find(*class);
        wire_by_class[*class] = wire_by_root[&root];
    }

    (wire_by_class, wires)
}

fn region_operation(region: &Region, path: &[usize]) -> Operation {
    format!(
        "region.{}.{}",
        region_path_label(path),
        region_kind_name(region.kind)
    )
    .parse()
    .expect("region operation name must be valid")
}

fn region_path_label(path: &[usize]) -> String {
    path.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

fn region_kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Data => "data",
        RegionKind::Control => "control",
        RegionKind::InterleavedControl => "interleaved-control",
        RegionKind::InterleavedData => "interleaved-data",
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

fn append_layer_trace(trace: &mut String, layer: &Layer, path: &[usize]) {
    let path_label = if path.is_empty() {
        "root".to_string()
    } else {
        region_path_label(path)
    };
    writeln!(
        trace,
        "\nlayer {path_label}: {} wires, {} operations, {} regions",
        layer.graph.h.w.0.len(),
        layer.graph.h.x.0.len(),
        layer.regions.len()
    )
    .expect("write to string cannot fail");

    if let Some(morphism) = &layer.morphism_to_parent {
        writeln!(trace, "  morphism boundary").expect("write to string cannot fail");
        for (child, parent) in morphism
            .boundary_relation
            .child_wires
            .iter()
            .zip(&morphism.boundary_relation.parent_wires)
        {
            writeln!(trace, "    child w{} ~ parent w{}", child.0, parent.0)
                .expect("write to string cannot fail");
        }
    }

    let data_context = DataRegionInterfaceContext::new(
        &layer.graph,
        &layer.regions,
        layer.morphism_to_parent.as_ref(),
    );
    for region in &layer.regions {
        let mut region_path = path.to_vec();
        region_path.push(region.index);
        let operation_names = region
            .operations
            .iter()
            .map(|operation_id| layer.graph.h.x.0[*operation_id].to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            trace,
            "  region {} {:?}: [{}]",
            region_path_label(&region_path),
            region.kind,
            operation_names
        )
        .expect("write to string cannot fail");

        if let Some(expansion) = &region.expansion {
            writeln!(trace, "    expands").expect("write to string cannot fail");
            append_layer_trace(trace, expansion, &region_path);
        } else {
            let interface = match region.kind {
                RegionKind::Data => data_context.data_region_interface(&layer.graph, region),
                RegionKind::Control => RegionInterface::control_region(&layer.graph, region),
                RegionKind::InterleavedControl | RegionKind::InterleavedData => {
                    panic!(
                        "interleaved regions must be expanded before becoming region graph operations"
                    )
                }
            };
            writeln!(
                trace,
                "    leaf interface: ({}) -> ({})",
                interface_wire_list(&interface.inputs),
                interface_wire_list(&interface.outputs)
            )
            .expect("write to string cannot fail");
            if !interface.equations.is_empty() {
                writeln!(trace, "    interface equations").expect("write to string cannot fail");
                for (left, right) in interface.equations {
                    writeln!(trace, "      w{left} ~ w{right}")
                        .expect("write to string cannot fail");
                }
            }
        }
    }
}

fn interface_wire_list(wires: &[InterfaceWire]) -> String {
    wires
        .iter()
        .map(|wire| match wire {
            InterfaceWire::Layer(wire) => format!("w{wire}"),
            InterfaceWire::Synthetic => "_".to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
