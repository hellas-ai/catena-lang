use std::{
    collections::{BTreeMap, HashMap},
    fmt::Write,
};

use crate::{
    compile::{
        analysis::{
            Layer,
            region_graph::{RegionGraph, RegionGraphRegion, region_graph_with_regions},
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    stdlib::operations::{OperationKind, actual_operation_kind, actual_operation_name},
    union_find::UnionFind,
};

pub(super) fn value_equivalence_trace(layer: &Layer) -> Vec<u8> {
    let region_graph = region_graph_with_regions(layer);
    let mut builder = ValueEquivalenceBuilder::default();
    builder.add_cfg_edge_equations(&region_graph);
    builder.add_monoidal_equations(&region_graph);
    builder.finish_trace().into_bytes()
}

#[derive(Default)]
struct ValueEquivalenceBuilder {
    term_ids: HashMap<ValueTerm, usize>,
    terms: Vec<ValueTerm>,
    equations: Vec<ValueEquation>,
}

impl ValueEquivalenceBuilder {
    fn add_cfg_edge_equations(&mut self, region_graph: &RegionGraph) {
        let connectivity = RegionGraphConnectivity::new(&region_graph.graph);
        for wire in connectivity.wires() {
            let Some((producer, output_index)) = connectivity.producer(wire) else {
                continue;
            };
            for (consumer, input_index) in connectivity.consumers(wire).iter().copied() {
                let Some(left) = region_output_term(&region_graph.regions[producer], output_index)
                else {
                    continue;
                };
                let Some(right) = region_input_term(&region_graph.regions[consumer], input_index)
                else {
                    continue;
                };
                self.add_equation(left, right, EquationReason::CfgEdge { wire });
            }
        }
    }

    fn add_monoidal_equations(&mut self, region_graph: &RegionGraph) {
        for region in &region_graph.regions {
            self.add_region_monoidal_equations(region);
        }
    }

    fn add_region_monoidal_equations(&mut self, region: &RegionGraphRegion) {
        for operation_id in &region.region.operations {
            let operation = operation_name(&region.graph, *operation_id);
            if actual_operation_kind(operation) != OperationKind::MonoidalStructure {
                continue;
            }

            let actual = actual_operation_name(operation);
            let inputs = operation_inputs(&region.graph, *operation_id)
                .map(|wire| wire.0)
                .collect::<Vec<_>>();
            let outputs = operation_outputs(&region.graph, *operation_id)
                .map(|wire| wire.0)
                .collect::<Vec<_>>();

            match actual {
                "val.*.intro" => {
                    self.add_product_intro(region, *operation_id, &inputs, &outputs);
                }
                "val.*.elim" => {
                    self.add_product_elim(region, *operation_id, &inputs, &outputs);
                }
                "unitl.intro" => {
                    self.add_unitl_intro(region, *operation_id, &inputs, &outputs);
                }
                "unitl.elim" => {
                    self.add_unitl_elim(region, *operation_id, &inputs, &outputs);
                }
                "val.+.intro" => {
                    self.add_sum_intro(region, *operation_id, &inputs, &outputs);
                }
                "val.+.elim" => {
                    self.add_sum_elim(region, *operation_id, &inputs, &outputs);
                }
                "2.intro" => {
                    self.add_two_intro(region, *operation_id, &inputs, &outputs);
                }
                "2.elim" => {
                    self.add_two_elim(region, *operation_id, &inputs, &outputs);
                }
                "distl" => {
                    self.add_distl(region, *operation_id, &inputs, &outputs);
                }
                "distr" => {
                    self.add_distr(region, *operation_id, &inputs, &outputs);
                }
                "elim2" => {
                    self.add_elim2(region, *operation_id, &inputs, &outputs);
                }
                _ => panic!(
                    "unknown monoidal structure operation {actual} in region.{} #{}",
                    path_label(&region.path),
                    operation_id
                ),
            }
        }
    }

    fn add_product_intro(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "val.*.intro", inputs, outputs, 2, 1);
        self.add_equation(
            term(&region.path, outputs[0]).field(0),
            term(&region.path, inputs[0]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "val.*.intro",
            },
        );
        self.add_equation(
            term(&region.path, outputs[0]).field(1),
            term(&region.path, inputs[1]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "val.*.intro",
            },
        );
    }

    fn add_product_elim(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "val.*.elim", inputs, outputs, 1, 2);
        self.add_equation(
            term(&region.path, inputs[0]).field(0),
            term(&region.path, outputs[0]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "val.*.elim",
            },
        );
        self.add_equation(
            term(&region.path, inputs[0]).field(1),
            term(&region.path, outputs[1]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "val.*.elim",
            },
        );
    }

    fn add_unitl_intro(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "unitl.intro", inputs, outputs, 1, 1);
        self.add_equation(
            term(&region.path, outputs[0]).field(1),
            term(&region.path, inputs[0]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "unitl.intro",
            },
        );
    }

    fn add_unitl_elim(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "unitl.elim", inputs, outputs, 1, 1);
        self.add_equation(
            term(&region.path, inputs[0]).field(1),
            term(&region.path, outputs[0]),
            EquationReason::Monoidal {
                region: region.path.clone(),
                operation_id,
                operation: "unitl.elim",
            },
        );
    }

    fn add_sum_intro(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "val.+.intro", inputs, outputs, 2, 1);

        for (branch, input) in inputs.iter().copied().enumerate() {
            self.add_equation(
                term(&region.path, outputs[0]).branch(branch),
                term(&region.path, input),
                monoidal_reason(region, operation_id, "val.+.intro"),
            );
        }
    }

    fn add_sum_elim(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "val.+.elim", inputs, outputs, 1, 2);

        for (branch, output) in outputs.iter().copied().enumerate() {
            self.add_equation(
                term(&region.path, inputs[0]).branch(branch),
                term(&region.path, output),
                monoidal_reason(region, operation_id, "val.+.elim"),
            );
        }
    }

    fn add_two_intro(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "2.intro", inputs, outputs, 1, 1);

        self.add_equation(
            term(&region.path, inputs[0]).tag(),
            term(&region.path, outputs[0]),
            monoidal_reason(region, operation_id, "2.intro"),
        );
    }

    fn add_two_elim(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "2.elim", inputs, outputs, 1, 1);

        self.add_equation(
            term(&region.path, outputs[0]).tag(),
            term(&region.path, inputs[0]),
            monoidal_reason(region, operation_id, "2.elim"),
        );
    }

    fn add_distl(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "distl", inputs, outputs, 1, 1);

        let input = term(&region.path, inputs[0]);
        let output = term(&region.path, outputs[0]);
        self.add_equation(
            output.clone().tag(),
            input.clone().field(1).tag(),
            monoidal_reason(region, operation_id, "distl"),
        );
        for branch in 0..2 {
            self.add_equation(
                output.clone().branch(branch).field(0),
                input.clone().field(0),
                monoidal_reason(region, operation_id, "distl"),
            );
            self.add_equation(
                output.clone().branch(branch).field(1),
                input.clone().field(1).branch(branch),
                monoidal_reason(region, operation_id, "distl"),
            );
        }
    }

    fn add_distr(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "distr", inputs, outputs, 1, 1);

        let input = term(&region.path, inputs[0]);
        let output = term(&region.path, outputs[0]);
        self.add_equation(
            output.clone().tag(),
            input.clone().field(0).tag(),
            monoidal_reason(region, operation_id, "distr"),
        );
        for branch in 0..2 {
            self.add_equation(
                output.clone().branch(branch).field(0),
                input.clone().field(0).branch(branch),
                monoidal_reason(region, operation_id, "distr"),
            );
            self.add_equation(
                output.clone().branch(branch).field(1),
                input.clone().field(1),
                monoidal_reason(region, operation_id, "distr"),
            );
        }
    }

    fn add_elim2(
        &mut self,
        region: &RegionGraphRegion,
        operation_id: usize,
        inputs: &[usize],
        outputs: &[usize],
    ) {
        assert_monoidal_arity(region, operation_id, "elim2", inputs, outputs, 1, 2);

        for (branch, output) in outputs.iter().copied().enumerate() {
            self.add_equation(
                term(&region.path, inputs[0]).branch(branch).field(1),
                term(&region.path, output),
                monoidal_reason(region, operation_id, "elim2"),
            );
        }
    }

    fn add_equation(&mut self, left: ValueTerm, right: ValueTerm, reason: EquationReason) {
        let left = self.term_id(left);
        let right = self.term_id(right);
        self.equations.push(ValueEquation {
            left,
            right,
            reason,
        });
    }

    fn term_id(&mut self, term: ValueTerm) -> usize {
        if let Some(id) = self.term_ids.get(&term) {
            return *id;
        }
        let id = self.terms.len();
        self.terms.push(term.clone());
        self.term_ids.insert(term, id);
        id
    }

    fn finish_trace(self) -> String {
        let mut union_find = UnionFind::new(self.terms.len());
        for equation in &self.equations {
            union_find.union(equation.left, equation.right);
        }

        let mut out = String::new();
        writeln!(&mut out, "# Value Equivalence\n").expect("write to string cannot fail");
        self.write_equations(&mut out);
        self.write_classes(&mut out, &mut union_find);
        out
    }

    fn write_equations(&self, out: &mut String) {
        writeln!(out, "equations").expect("write to string cannot fail");
        for equation in &self.equations {
            writeln!(
                out,
                "  {} ~ {}    {}",
                self.terms[equation.left], self.terms[equation.right], equation.reason
            )
            .expect("write to string cannot fail");
        }
    }

    fn write_classes(&self, out: &mut String, union_find: &mut UnionFind) {
        let mut classes = BTreeMap::<usize, Vec<&ValueTerm>>::new();
        for (id, term) in self.terms.iter().enumerate() {
            classes.entry(union_find.find(id)).or_default().push(term);
        }

        writeln!(out, "\nclasses").expect("write to string cannot fail");
        for terms in classes.values() {
            if terms.len() < 2 {
                continue;
            }
            let rendered = terms.iter().map(ToString::to_string).collect::<Vec<_>>();
            writeln!(out, "  {}", rendered.join(" = ")).expect("write to string cannot fail");
        }
    }
}

#[derive(Debug, Clone)]
struct ValueEquation {
    left: usize,
    right: usize,
    reason: EquationReason,
}

#[derive(Debug, Clone)]
enum EquationReason {
    CfgEdge {
        wire: usize,
    },
    Monoidal {
        region: Vec<usize>,
        operation_id: usize,
        operation: &'static str,
    },
}

impl std::fmt::Display for EquationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CfgEdge { wire } => write!(f, "[cfg edge w{wire}]"),
            Self::Monoidal {
                region,
                operation_id,
                operation,
            } => write!(
                f,
                "[region.{} #{operation_id} {operation}]",
                path_label(region)
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ValueTerm {
    region: Vec<usize>,
    wire: usize,
    path: Vec<ValuePathStep>,
}

impl ValueTerm {
    fn field(mut self, field: usize) -> Self {
        self.path.push(ValuePathStep::Product(field));
        self
    }

    fn branch(mut self, branch: usize) -> Self {
        self.path.push(ValuePathStep::Sum(branch));
        self
    }

    fn tag(mut self) -> Self {
        self.path.push(ValuePathStep::Tag);
        self
    }
}

impl std::fmt::Display for ValueTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "region.{}:w{}", path_label(&self.region), self.wire)?;
        for step in &self.path {
            match step {
                ValuePathStep::Product(field) => write!(f, ".{field}")?,
                ValuePathStep::Sum(branch) => write!(f, ".case{branch}")?,
                ValuePathStep::Tag => write!(f, ".tag")?,
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ValuePathStep {
    Product(usize),
    Sum(usize),
    Tag,
}

fn term(region: &[usize], wire: usize) -> ValueTerm {
    ValueTerm {
        region: region.to_vec(),
        wire,
        path: Vec::new(),
    }
}

fn monoidal_reason(
    region: &RegionGraphRegion,
    operation_id: usize,
    operation: &'static str,
) -> EquationReason {
    EquationReason::Monoidal {
        region: region.path.clone(),
        operation_id,
        operation,
    }
}

fn assert_monoidal_arity(
    region: &RegionGraphRegion,
    operation_id: usize,
    operation: &str,
    inputs: &[usize],
    outputs: &[usize],
    expected_inputs: usize,
    expected_outputs: usize,
) {
    assert!(
        inputs.len() == expected_inputs && outputs.len() == expected_outputs,
        "unexpected arity for monoidal operation {operation} in region.{} #{}: expected {} -> {}, got {} -> {}",
        path_label(&region.path),
        operation_id,
        expected_inputs,
        expected_outputs,
        inputs.len(),
        outputs.len()
    );
}

fn region_input_term(region: &RegionGraphRegion, input_index: usize) -> Option<ValueTerm> {
    region
        .inputs
        .get(input_index)
        .map(|wire| term(&region.path, *wire))
}

fn region_output_term(region: &RegionGraphRegion, output_index: usize) -> Option<ValueTerm> {
    region
        .outputs
        .get(output_index)
        .map(|wire| term(&region.path, *wire))
}

fn path_label(path: &[usize]) -> String {
    path.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

struct RegionGraphConnectivity {
    consumers_by_wire: HashMap<usize, Vec<(usize, usize)>>,
    producer_by_wire: HashMap<usize, (usize, usize)>,
}

impl RegionGraphConnectivity {
    fn new(graph: &Graph) -> Self {
        let mut consumers_by_wire = HashMap::<usize, Vec<(usize, usize)>>::new();
        let mut producer_by_wire = HashMap::<usize, (usize, usize)>::new();

        for operation_id in 0..graph.h.x.0.len() {
            for (input_index, wire) in operation_inputs(graph, operation_id).enumerate() {
                consumers_by_wire
                    .entry(wire.0)
                    .or_default()
                    .push((operation_id, input_index));
            }

            for (output_index, wire) in operation_outputs(graph, operation_id).enumerate() {
                let previous = producer_by_wire.insert(wire.0, (operation_id, output_index));
                assert!(
                    previous.is_none(),
                    "region graph wire w{} has multiple producers",
                    wire.0
                );
            }
        }

        Self {
            consumers_by_wire,
            producer_by_wire,
        }
    }

    fn wires(&self) -> Vec<usize> {
        let mut wires = self
            .producer_by_wire
            .keys()
            .chain(self.consumers_by_wire.keys())
            .copied()
            .collect::<Vec<_>>();
        wires.sort_unstable();
        wires.dedup();
        wires
    }

    fn producer(&self, wire: usize) -> Option<(usize, usize)> {
        self.producer_by_wire.get(&wire).copied()
    }

    fn consumers(&self, wire: usize) -> &[(usize, usize)] {
        self.consumers_by_wire
            .get(&wire)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}
