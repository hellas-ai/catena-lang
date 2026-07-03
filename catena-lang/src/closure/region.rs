//! Identify "closure regions" in a term.
//!
//!
use std::collections::{HashMap, VecDeque};

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{EdgeId, NodeId};
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    prefixes::NAME_PREFIX,
    stdlib::constants::{DEFER, FN_HOM_TYPE},
};

pub type Obj = Tree<(), Operation>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureRegion {
    pub closure_wire: NodeId,
    pub closure_type: Obj,
    pub leaf_inputs: Vec<NodeId>,
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
/// operation: `defer` or `name.*`. Sources of those leaf operations are
/// retained as closure environment inputs.
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
    let leaf_inputs = leaf_inputs(definition, &edges);
    Ok(ClosureRegion {
        closure_wire,
        closure_type: closure_type.clone(),
        leaf_inputs,
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

fn leaf_inputs(definition: &AnnotatedTerm, edges: &[EdgeId]) -> Vec<NodeId> {
    edges
        .iter()
        .filter(|edge| is_region_leaf(&definition.hypergraph.edges[edge.0]))
        .flat_map(|edge| {
            definition.hypergraph.adjacency[edge.0]
                .sources
                .iter()
                .copied()
        })
        .collect()
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == FN_HOM_TYPE && children.len() == 2
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
