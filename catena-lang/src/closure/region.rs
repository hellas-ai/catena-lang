//! Identify "closure regions" in a term.
//!
//!
use std::collections::{HashMap, VecDeque};

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::check::AnnotatedTerm;

pub type Obj = Tree<(), Operation>;

const CLOSURE_TYPE: &str = "=>";
const DEFER: &str = "defer";
const NAME_PREFIX: &str = "name.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRegion {
    pub closure_wire: NodeId,
    pub closure_type: Obj,
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
}

#[derive(Debug, Error)]
pub enum ClosureRegionError {
    #[error("closure region root node n{wire} is out of bounds")]
    WireOutOfBounds { wire: usize },
    #[error("closure region root node n{wire} is not closure-typed")]
    NotClosureTyped { wire: usize },
    #[error("closure region root node n{wire} has no producer")]
    UnproducedClosureWire { wire: usize },
}

/// Find closure-construction regions rooted at the requested closure wires.
///
/// Each `closure_wires` entry must name a closure-typed node in `definition`.
/// The result order matches the input wire order. For each root, the region is
/// found by walking left through producer edges until reaching an included leaf
/// operation: `defer` or `name.*`.
pub fn closure_region(
    definition: &AnnotatedTerm,
    closure_wires: &[NodeId],
) -> Result<Vec<ClosureRegion>, ClosureRegionError> {
    let connectivity = Connectivity::new(definition);
    closure_wires
        .iter()
        .copied()
        .map(|closure_wire| {
            closure_region_with_connectivity(definition, &connectivity, closure_wire)
        })
        .collect()
}

// Find a ClosureRegion by searching "leftwards" from a NodeId within an AnnotatedTerm, using
// Connectivity as an index to speed up search.
fn closure_region_with_connectivity(
    definition: &AnnotatedTerm,
    connectivity: &Connectivity,
    closure_wire: NodeId,
) -> Result<ClosureRegion, ClosureRegionError> {
    let closure_type = definition.hypergraph.nodes.get(closure_wire.0).ok_or(
        ClosureRegionError::WireOutOfBounds {
            wire: closure_wire.0,
        },
    )?;
    if !is_closure_type(closure_type) {
        return Err(ClosureRegionError::NotClosureTyped {
            wire: closure_wire.0,
        });
    }

    let Region { nodes, edges } = build_closure_region(definition, &connectivity, closure_wire)?;
    Ok(ClosureRegion {
        closure_wire,
        closure_type: closure_type.clone(),
        nodes,
        edges,
    })
}

// Search leftwards from a closure NodeId until we find any terminal edge: name or defer (see
// is_region_leaf).
fn build_closure_region(
    definition: &AnnotatedTerm,
    connectivity: &Connectivity,
    closure_wire: NodeId,
) -> Result<Region, ClosureRegionError> {
    let Some(&producer) = connectivity.producer_by_wire.get(&closure_wire.0) else {
        return Err(ClosureRegionError::UnproducedClosureWire {
            wire: closure_wire.0,
        });
    };

    let mut region = RegionBuilder::new(definition);
    let mut pending = VecDeque::from([producer]);

    while let Some(edge_id) = pending.pop_front() {
        if !region.insert_edge(edge_id) {
            continue;
        }

        let operation = &definition.hypergraph.edges[edge_id.0];
        let hyperedge = &definition.hypergraph.adjacency[edge_id.0];
        region.insert_nodes(hyperedge.sources.iter().copied());
        region.insert_nodes(hyperedge.targets.iter().copied());

        if is_region_leaf(operation) {
            continue;
        }

        for source in &hyperedge.sources {
            if let Some(&source_producer) = connectivity.producer_by_wire.get(&source.0) {
                pending.push_back(source_producer);
            }
        }
    }

    Ok(region.finish())
}

fn is_region_leaf(operation: &Operation) -> bool {
    operation.as_str() == DEFER || operation.as_str().starts_with(NAME_PREFIX)
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == CLOSURE_TYPE && children.len() == 2
}

struct Connectivity {
    producer_by_wire: HashMap<usize, EdgeId>,
}

impl Connectivity {
    fn new(definition: &AnnotatedTerm) -> Self {
        let mut producer_by_wire = HashMap::new();
        for (edge_index, hyperedge) in definition.hypergraph.adjacency.iter().enumerate() {
            for target in &hyperedge.targets {
                producer_by_wire.insert(target.0, EdgeId(edge_index));
            }
        }
        Self { producer_by_wire }
    }
}

struct Region {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

struct RegionBuilder {
    nodes: Vec<bool>,
    edges: Vec<bool>,
}

impl RegionBuilder {
    fn new(definition: &AnnotatedTerm) -> Self {
        Self {
            nodes: vec![false; definition.hypergraph.nodes.len()],
            edges: vec![false; definition.hypergraph.edges.len()],
        }
    }

    fn insert_edge(&mut self, edge_id: EdgeId) -> bool {
        let already_present = self.edges[edge_id.0];
        self.edges[edge_id.0] = true;
        !already_present
    }

    fn insert_nodes(&mut self, nodes: impl IntoIterator<Item = NodeId>) {
        for node in nodes {
            self.nodes[node.0] = true;
        }
    }

    fn finish(self) -> Region {
        let nodes = self
            .nodes
            .into_iter()
            .enumerate()
            .filter_map(|(index, present)| present.then_some(NodeId(index)))
            .collect();
        let edges = self
            .edges
            .into_iter()
            .enumerate()
            .filter_map(|(index, present)| present.then_some(EdgeId(index)))
            .collect();
        Region { nodes, edges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        check::{DefinitionTypes, check},
        elaborate::elaborate,
        stdlib,
    };
    use metacat::theory::{RawTheorySet, Theory, TheoryId, TheorySet};

    #[test]
    fn identity_wire_has_no_closure_regions() {
        let definition = annotated_program_definition(
            r#"
            (def program id : [a] -> [a] = [x])
            "#,
            "id",
        );
        let closure_wires = closure_wires(&definition);
        let regions =
            closure_region(&definition, &closure_wires).expect("region discovery should succeed");

        assert_eq!(regions.len(), 0);
        assert_eq!(
            regions
                .iter()
                .map(|region| region.edges.len())
                .sum::<usize>(),
            0
        );
    }

    #[test]
    fn deferred_input_composed_with_lifted_bool_id_has_closure_regions() {
        // Create a simple program that consists of one region plus some parts before it
        // -- the parts before correspond roughly to ¬(x ∧ T)
        let definition = annotated_program_definition(
            r#"
            (def program run-bool-id : (bool val) -> ({1 (bool val)} =>) = (
              {[x] bool.t}
              bool.and
              bool.not
              {defer (name.bool.id lift)}
              compose
            ))
            "#,
            "run-bool-id",
        );

        let node_id = definition.targets[0];
        let regions =
            closure_region(&definition, &[node_id]).expect("region discovery should succeed");

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].edges.len(), 4);
    }

    fn theories_with(source: &'static str) -> (TheorySet, DefinitionTypes) {
        let raw_theories = RawTheorySet::from_texts(stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = elaborate(raw_theories).expect("test theories should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theories should load");
        let definition_types = check(&theory_set).expect("test theories should typecheck");
        (theory_set, definition_types)
    }

    fn annotated_program_definition(source: &'static str, definition: &str) -> AnnotatedTerm {
        let (theory_set, definition_types) = theories_with(source);
        let program = TheoryId("program".parse().expect("program theory id should parse"));
        let definition: Operation = definition
            .parse()
            .expect("program definition name should parse");
        let theory = theory_set
            .theories
            .get(&program)
            .expect("program theory should exist");
        let Theory::Theory { arrows, .. } = theory else {
            panic!("program should be a theory");
        };
        let arrow = arrows
            .get(&definition)
            .expect("program definition should exist");
        let mut body = arrow
            .definition
            .clone()
            .expect("program arrow should be a definition");
        body.quotient().ok();
        let labels = definition_types
            .get(&program)
            .and_then(|definitions| definitions.get(&definition))
            .cloned()
            .expect("program definition should have checked node types");
        body.with_nodes(|_| labels)
            .expect("checked node labels should match definition graph")
    }

    fn closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
        definition
            .hypergraph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, object)| is_closure_type(object).then_some(NodeId(index)))
            .collect()
    }
}
