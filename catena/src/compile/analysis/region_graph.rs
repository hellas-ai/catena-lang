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
};

pub(super) fn region_graph(layer: &Layer) -> Graph {
    let mut builder = RegionGraphBuilder::default();
    builder.add_layer(layer);
    builder.finish()
}

#[derive(Default)]
struct RegionGraphBuilder {
    wires: Vec<Obj>,
    operations: Vec<Operation>,
    source_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    source_values: Vec<usize>,
    target_values: Vec<usize>,
}

impl RegionGraphBuilder {
    fn add_layer(&mut self, layer: &Layer) {
        for region in &layer.regions {
            if let Some(expansion) = &region.expansion {
                self.add_layer(expansion);
                continue;
            }

            let interface = RegionInterface::new(&layer.graph, region);
            self.add_region_operation(region, interface);
        }
    }

    fn add_region_operation(&mut self, region: &Region, interface: RegionInterface) {
        let wire_base = self.wires.len();
        let source_wires = (0..interface.inputs)
            .map(|wire| wire_base + wire)
            .collect::<Vec<_>>();
        let target_wires = (0..interface.outputs)
            .map(|wire| wire_base + interface.inputs + wire)
            .collect::<Vec<_>>();

        self.wires
            .extend((0..interface.inputs + interface.outputs).map(|_| Tree::Empty));
        self.operations.push(region_operation(region));
        self.source_lengths.push(source_wires.len());
        self.target_lengths.push(target_wires.len());
        self.source_values.extend(source_wires);
        self.target_values.extend(target_wires);
    }

    fn finish(self) -> Graph {
        let wire_count = self.wires.len();
        OpenHypergraph {
            s: finite_function(Vec::new(), wire_count),
            t: finite_function(Vec::new(), wire_count),
            h: Hypergraph {
                s: indexed_coproduct(self.source_lengths, self.source_values, wire_count),
                t: indexed_coproduct(self.target_lengths, self.target_values, wire_count),
                w: SemifiniteFunction::new(VecArray(self.wires)),
                x: SemifiniteFunction::new(VecArray(self.operations)),
            },
        }
        .validate()
        .expect("region graph must be valid")
    }
}

#[derive(Debug, Clone, Copy)]
struct RegionInterface {
    inputs: usize,
    outputs: usize,
}

impl RegionInterface {
    fn new(graph: &Graph, region: &Region) -> Self {
        match region.kind {
            RegionKind::Data => Self {
                inputs: 1,
                outputs: 1,
            },
            RegionKind::Control => Self::native_control(graph, region),
            RegionKind::InterleavedControl | RegionKind::InterleavedData => {
                panic!("expanded regions must not become region-graph operations")
            }
        }
    }

    fn native_control(graph: &Graph, region: &Region) -> Self {
        let boundary = RegionBoundary::new(graph, region);
        match (boundary.inputs.len(), boundary.outputs.len()) {
            (1, 1) | (1, 2) | (2, 1) => Self {
                inputs: boundary.inputs.len(),
                outputs: boundary.outputs.len(),
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
struct RegionBoundary {
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl RegionBoundary {
    fn new(graph: &Graph, region: &Region) -> Self {
        let consumed = region_consumed_wires(graph, region);
        let produced = region_produced_wires(graph, region);
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

fn region_operation(region: &Region) -> Operation {
    format!(
        "analysis.region.{}.{}",
        region_kind_name(region.kind),
        region.index
    )
    .parse()
    .expect("region operation name must be valid")
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
