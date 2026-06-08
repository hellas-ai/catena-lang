use std::{
    collections::{BTreeMap, HashMap},
    fmt::Write,
};

use crate::{
    compile::{
        cfg::{
            layering::Layer,
            region_graph::{RegionGraph, RegionGraphRegion, lower_layer_to_region_graph},
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    stdlib::operations::{OperationKind, actual_operation_kind, actual_operation_name},
    union_find::UnionFind,
};

pub(super) fn value_equivalence_trace(layer: &Layer) -> Vec<u8> {
    let region_graph = lower_layer_to_region_graph(layer);
    let equivalences = compute_value_equivalences(&region_graph);
    equivalences.trace().into_bytes()
}

pub(super) fn compute_value_equivalences(region_graph: &RegionGraph) -> ValueEquivalences {
    let mut builder = ValueEquivalenceBuilder::default();
    builder.add_observed_terms(region_graph);
    builder.add_cfg_edge_equations(region_graph);
    builder.add_monoidal_equations(region_graph);
    builder.finish()
}

#[derive(Debug, Clone)]
pub(super) struct ValueEquivalences {
    terms: Vec<ValueTerm>,
    equations: Vec<ValueEquation>,
    class_by_term: HashMap<ValueTerm, usize>,
    // Equivalence classes are semantic values spanning region-local wire
    // namespaces. CFG variable ids must therefore be allocated per class; using
    // a representative term's local wire number would collide across regions.
    variable_by_class: HashMap<usize, usize>,
}

impl ValueEquivalences {
    pub(super) fn resolve_wire(&self, region: &[usize], wire: usize) -> usize {
        self.resolve(region, wire, &[])
    }

    pub(super) fn resolve(&self, region: &[usize], wire: usize, path: &[ValueProjection]) -> usize {
        let term = ValueTerm {
            region: region.to_vec(),
            wire,
            path: path
                .iter()
                .copied()
                .map(ValuePathStep::from)
                .collect::<Vec<_>>(),
        };
        self.class_by_term
            .get(&term)
            .and_then(|class| self.variable_by_class.get(class))
            .copied()
            .unwrap_or(wire)
    }

    fn trace(&self) -> String {
        let mut out = String::new();
        writeln!(&mut out, "# Value Equivalence\n").expect("write to string cannot fail");
        self.write_equations(&mut out);
        self.write_classes(&mut out);
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

    fn write_classes(&self, out: &mut String) {
        let mut classes = BTreeMap::<usize, Vec<&ValueTerm>>::new();
        for term in &self.terms {
            if let Some(class) = self.class_by_term.get(term).copied() {
                classes.entry(class).or_default().push(term);
            }
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

#[derive(Debug, Clone, Copy)]
pub(super) enum ValueProjection {
    Product(usize),
    Tag,
}

impl From<ValueProjection> for ValuePathStep {
    fn from(value: ValueProjection) -> Self {
        match value {
            ValueProjection::Product(field) => Self::Product(field),
            ValueProjection::Tag => Self::Tag,
        }
    }
}

#[derive(Default)]
struct ValueEquivalenceBuilder {
    term_ids: HashMap<ValueTerm, usize>,
    terms: Vec<ValueTerm>,
    equations: Vec<ValueEquation>,
}

impl ValueEquivalenceBuilder {
    fn add_observed_terms(&mut self, region_graph: &RegionGraph) {
        for region in &region_graph.regions {
            for wire in region.inputs.iter().chain(&region.outputs) {
                self.term_id(term(&region.path, *wire));
            }
            for operation_id in &region.region.operations {
                for wire in operation_inputs(&region.graph, *operation_id)
                    .chain(operation_outputs(&region.graph, *operation_id))
                {
                    self.term_id(term(&region.path, wire.0));
                }
            }
        }
    }

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

    fn finish(mut self) -> ValueEquivalences {
        self.add_base_terms_for_observed_paths();
        self.close_under_congruence();
        let mut union_find = UnionFind::new(self.terms.len());
        for equation in &self.equations {
            union_find.union(equation.left, equation.right);
        }

        self.add_congruence_equations(&mut union_find);
        let mut class_by_term = HashMap::new();
        for (id, term) in self.terms.iter().enumerate() {
            class_by_term.insert(term.clone(), union_find.find(id));
        }

        let mut terms_by_class = BTreeMap::<usize, Vec<&ValueTerm>>::new();
        for term in &self.terms {
            terms_by_class
                .entry(class_by_term[term])
                .or_default()
                .push(term);
        }
        let mut used_variables = std::collections::HashSet::new();
        let mut next_variable = self.terms.iter().map(|term| term.wire).max().unwrap_or(0) + 1;
        let mut variable_by_class = HashMap::new();
        for (class, terms) in terms_by_class {
            let variable = preferred_root_variable(&terms)
                .filter(|variable| used_variables.insert(*variable))
                .unwrap_or_else(|| {
                    while used_variables.contains(&next_variable) {
                        next_variable += 1;
                    }
                    let variable = next_variable;
                    used_variables.insert(variable);
                    next_variable += 1;
                    variable
                });
            variable_by_class.insert(class, variable);
        }

        ValueEquivalences {
            terms: self.terms,
            equations: self.equations,
            class_by_term,
            variable_by_class,
        }
    }

    fn close_under_congruence(&mut self) {
        loop {
            let mut union_find = UnionFind::new(self.terms.len());
            for equation in &self.equations {
                union_find.union(equation.left, equation.right);
            }
            self.add_congruence_equations(&mut union_find);

            let additions = self.congruence_projection_additions(&mut union_find);
            if additions.is_empty() {
                break;
            }

            for (left, right) in additions {
                self.add_equation(left, right, EquationReason::Congruence);
            }
        }
    }

    fn add_base_terms_for_observed_paths(&mut self) {
        let base_terms = self
            .terms
            .iter()
            .filter(|term| !term.path.is_empty())
            .map(|term| ValueTerm {
                region: term.region.clone(),
                wire: term.wire,
                path: Vec::new(),
            })
            .collect::<Vec<_>>();
        for term in base_terms {
            self.term_id(term);
        }
    }

    fn add_congruence_equations(&self, union_find: &mut UnionFind) {
        let mut terms_by_prefix_and_suffix =
            BTreeMap::<(usize, Vec<ValuePathStep>), Vec<usize>>::new();
        for (id, term) in self.terms.iter().enumerate() {
            for split in 0..term.path.len() {
                let prefix = ValueTerm {
                    region: term.region.clone(),
                    wire: term.wire,
                    path: term.path[..split].to_vec(),
                };
                let Some(prefix_id) = self.term_ids.get(&prefix).copied() else {
                    continue;
                };
                terms_by_prefix_and_suffix
                    .entry((union_find.find(prefix_id), term.path[split..].to_vec()))
                    .or_default()
                    .push(id);
            }
        }

        for ids in terms_by_prefix_and_suffix.values() {
            if let Some(first) = ids.first().copied() {
                for id in ids.iter().copied().skip(1) {
                    union_find.union(first, id);
                }
            }
        }
    }

    fn congruence_projection_additions(
        &self,
        union_find: &mut UnionFind,
    ) -> Vec<(ValueTerm, ValueTerm)> {
        let mut prefixes_by_class = BTreeMap::<usize, Vec<ValueTerm>>::new();
        for (id, term) in self.terms.iter().enumerate() {
            prefixes_by_class
                .entry(union_find.find(id))
                .or_default()
                .push(term.clone());
        }

        let mut projected_by_prefix_class_and_suffix =
            BTreeMap::<(usize, Vec<ValuePathStep>), ValueTerm>::new();
        for term in &self.terms {
            for split in 0..term.path.len() {
                let prefix = ValueTerm {
                    region: term.region.clone(),
                    wire: term.wire,
                    path: term.path[..split].to_vec(),
                };
                let Some(prefix_id) = self.term_ids.get(&prefix).copied() else {
                    continue;
                };
                let prefix_class = union_find.find(prefix_id);
                projected_by_prefix_class_and_suffix
                    .entry((prefix_class, term.path[split..].to_vec()))
                    .or_insert_with(|| term.clone());
            }
        }

        let mut additions = Vec::new();
        for ((prefix_class, suffix), representative) in projected_by_prefix_class_and_suffix {
            let Some(prefixes) = prefixes_by_class.get(&prefix_class) else {
                continue;
            };
            for prefix in prefixes {
                let projected = ValueTerm {
                    region: prefix.region.clone(),
                    wire: prefix.wire,
                    path: prefix
                        .path
                        .iter()
                        .copied()
                        .chain(suffix.iter().copied())
                        .collect(),
                };
                if !self.term_ids.contains_key(&projected) {
                    additions.push((projected, representative.clone()));
                }
            }
        }
        additions
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
    Congruence,
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
            Self::Congruence => write!(f, "[congruence]"),
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

fn preferred_root_variable(terms: &[&ValueTerm]) -> Option<usize> {
    // Region-local wires are not globally unique. Keeping a root-layer base
    // wire when available preserves source/ABI names, but every other class
    // must receive a fresh CFG variable id.
    terms
        .iter()
        .filter(|term| term.path.is_empty() && term.region.len() == 1)
        .min_by_key(|term| (term.region.as_slice(), term.wire))
        .map(|term| term.wire)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_region_classes_do_not_share_local_wire_numbers() {
        let mut builder = ValueEquivalenceBuilder::default();
        builder.term_id(term(&[0], 6));
        builder.term_id(term(&[1, 2, 0], 6));

        let equivalences = builder.finish();

        assert_eq!(equivalences.resolve_wire(&[0], 6), 6);
        assert_ne!(equivalences.resolve_wire(&[1, 2, 0], 6), 6);
    }

    #[test]
    fn local_singleton_terms_receive_fresh_cfg_variables() {
        let mut builder = ValueEquivalenceBuilder::default();
        builder.term_id(term(&[1, 2, 0], 4));

        let equivalences = builder.finish();

        assert_ne!(equivalences.resolve_wire(&[1, 2, 0], 4), 4);
    }

    #[test]
    fn projected_equivalence_uses_one_fresh_cfg_variable() {
        let mut builder = ValueEquivalenceBuilder::default();
        builder.add_equation(
            term(&[1, 2, 0], 9).tag(),
            term(&[1, 2, 0], 6),
            EquationReason::Congruence,
        );

        let equivalences = builder.finish();
        let tag = equivalences.resolve(&[1, 2, 0], 9, &[ValueProjection::Tag]);
        let output = equivalences.resolve_wire(&[1, 2, 0], 6);

        assert_eq!(tag, output);
        assert_ne!(output, 6);
    }
}
