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
        analysis::{Layer, Region, partition::RegionKind},
        graph_ops::{Graph, operation_inputs, operation_outputs},
    },
    lang::Obj,
    union_find::UnionFind,
};

pub(super) fn region_graph(layer: &Layer) -> Graph {
    let mut builder = RegionGraphBuilder::default();
    builder.add_layer(layer);
    builder.finish()
}

pub(super) fn region_graph_trace(layer: &Layer) -> Vec<u8> {
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
}

impl RegionGraphBuilder {
    fn add_layer(&mut self, layer: &Layer) {
        self.add_nested_layer(layer, None, Vec::new());
    }

    fn add_nested_layer(&mut self, layer: &Layer, parent_layer: Option<usize>, path: Vec<usize>) {
        let layer_id = self.alloc_layer(layer);
        self.connect_to_parent(layer, layer_id, parent_layer);

        let data_context = DataRegionInterfaceContext::new(
            &layer.graph,
            &layer.regions,
            layer.morphism_to_parent.as_ref(),
        );
        for region in &layer.regions {
            let mut region_path = path.clone();
            region_path.push(region.index);

            if let Some(expansion) = &region.expansion {
                self.add_nested_layer(expansion, Some(layer_id), region_path);
                continue;
            }

            let interface = RegionInterface::new(&layer.graph, &data_context, region);
            self.add_region_operation(layer_id, region, region_path, interface);
        }
    }

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

    fn connect_to_parent(&mut self, layer: &Layer, layer_id: usize, parent_layer: Option<usize>) {
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
            let child_class = self.layer_wire_class(layer_id, child_wire);
            let parent_class = if multi_fiber_parent_wires.contains(&fiber_point.parent_wire.0) {
                self.boundary_fiber_class(
                    parent_layer,
                    fiber_point.parent_wire.0,
                    fiber_point.fiber_position,
                )
            } else {
                self.layer_wire_class(parent_layer, fiber_point.parent_wire.0)
            };
            self.equations.push((child_class, parent_class));
        }
    }

    fn add_region_operation(
        &mut self,
        layer_id: usize,
        region: &Region,
        path: Vec<usize>,
        interface: RegionInterface,
    ) {
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

    fn finish(self) -> Graph {
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
        OpenHypergraph {
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
        .expect("region graph must be valid")
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
    fn new(graph: &Graph, data_context: &DataRegionInterfaceContext, region: &Region) -> Self {
        match region.kind {
            RegionKind::Data => data_context.data_region_interface(graph, region),
            RegionKind::Control => Self::control_region(graph, region),
            RegionKind::InterleavedControl | RegionKind::InterleavedData => {
                panic!(
                    "interleaved regions must be expanded before becoming region graph operations"
                )
            }
        }
    }

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
        morphism_to_parent: Option<&crate::compile::analysis::NestingMorphism>,
    ) -> Self {
        let control_operations = regions
            .iter()
            .filter(|region| matches!(region.kind, RegionKind::InterleavedControl))
            .flat_map(|region| region.operations.iter().copied())
            .collect::<Vec<_>>();

        let graph_sources = morphism_to_parent
            .map(|morphism| {
                morphism
                    .boundary_relation
                    .child_wires_on_side(crate::compile::analysis::layering::BoundarySide::Source)
                    .into_iter()
                    .map(|wire| wire.0)
                    .collect()
            })
            .unwrap_or_default();
        let graph_targets = morphism_to_parent
            .map(|morphism| {
                morphism
                    .boundary_relation
                    .child_wires_on_side(crate::compile::analysis::layering::BoundarySide::Target)
                    .into_iter()
                    .map(|wire| wire.0)
                    .collect()
            })
            .unwrap_or_default();

        Self {
            graph_sources,
            graph_targets,
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

    fn data_region_interface(&self, graph: &Graph, region: &Region) -> RegionInterface {
        let boundary = self.data_region_boundary(graph, region);
        let input = boundary.inputs.first().copied();
        let output = boundary.outputs.first().copied();
        let equations = boundary
            .inputs
            .iter()
            .copied()
            .skip(1)
            .map(|wire| (input.expect("non-empty input boundary"), wire))
            .chain(
                boundary
                    .outputs
                    .iter()
                    .copied()
                    .skip(1)
                    .map(|wire| (output.expect("non-empty output boundary"), wire)),
            )
            .collect();
        RegionInterface {
            inputs: vec![
                input
                    .map(InterfaceWire::Layer)
                    .unwrap_or(InterfaceWire::Synthetic),
            ],
            outputs: vec![
                output
                    .map(InterfaceWire::Layer)
                    .unwrap_or(InterfaceWire::Synthetic),
            ],
            equations,
        }
    }

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

fn multi_fiber_parent_wires(
    fiber_points: &[Option<crate::compile::analysis::layering::BoundaryFiberPoint>],
) -> HashSet<usize> {
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
            let interface = RegionInterface::new(&layer.graph, &data_context, region);
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
