use std::collections::HashSet;

use hexpr::Operation;
use open_hypergraphs::{
    array::vec::VecArray,
    semifinite::SemifiniteFunction,
    strict::vec::{FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph},
};
use thiserror::Error;

use crate::lang::Obj;

pub type HostHypergraph = Hypergraph<Obj, Operation>;
pub type OperationId = usize;
pub type WireId = usize;

#[derive(Debug, Clone)]
pub struct Subgraph {
    pub graph: HostHypergraph,
    pub embedding: SubgraphEmbedding,
}

impl Subgraph {
    pub fn open_graph(&self) -> OpenHypergraph<Obj, Operation> {
        let wire_count = self.graph.w.0.len();
        OpenHypergraph {
            s: FiniteFunction::new(VecArray(Vec::new()), wire_count)
                .expect("empty source boundary is valid"),
            t: FiniteFunction::new(VecArray(Vec::new()), wire_count)
                .expect("empty target boundary is valid"),
            h: self.graph.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubgraphEmbedding {
    /// Local subgraph wire id -> host wire id.
    pub wires: FiniteFunction,
    /// Local subgraph operation id -> host operation id.
    pub operations: FiniteFunction,
}

#[derive(Debug, Error)]
pub enum SubgraphError {
    #[error("subgraph wire embedding is not injective")]
    NonInjectiveWireEmbedding,
    #[error("subgraph operation embedding is not injective")]
    NonInjectiveOperationEmbedding,
    #[error("wire id {wire} is out of bounds for host with {wire_count} wires")]
    WireOutOfBounds { wire: WireId, wire_count: usize },
    #[error("operation id {operation} is out of bounds for host with {operation_count} operations")]
    OperationOutOfBounds {
        operation: OperationId,
        operation_count: usize,
    },
    #[error("subgraph wire embedding is missing incident host wire {wire}")]
    MissingIncidentWire { wire: WireId },
    #[error("cannot construct finite function")]
    InvalidFiniteFunction,
    #[error("cannot construct indexed coproduct")]
    InvalidIndexedCoproduct,
    #[error("cannot glue subgraphs with different host wire targets")]
    WireTargetMismatch,
    #[error("cannot glue subgraphs with different host operation targets")]
    OperationTargetMismatch,
}

pub fn subgraph_from_operations(
    host: &HostHypergraph,
    operations: impl IntoIterator<Item = OperationId>,
) -> Result<Subgraph, SubgraphError> {
    let operations =
        embedding_from_unique_values(operations, operation_count(host), |operation| {
            SubgraphError::OperationOutOfBounds {
                operation,
                operation_count: operation_count(host),
            }
        })?;
    let wires = incident_wires(host, &operations)?;
    subgraph_from_embedding(host, SubgraphEmbedding { wires, operations })
}

pub fn subgraph_from_embedding(
    host: &HostHypergraph,
    embedding: SubgraphEmbedding,
) -> Result<Subgraph, SubgraphError> {
    validate_embedding(host, &embedding)?;

    let wire_inverse = inverse_on_image(&embedding.wires)?;
    let s = host
        .s
        .map_indexes(&embedding.operations)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;
    let t = host
        .t
        .map_indexes(&embedding.operations)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;

    ensure_incidence_is_in_image(&s, &embedding.wires)?;
    ensure_incidence_is_in_image(&t, &embedding.wires)?;

    let s = s
        .map_values(&wire_inverse)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;
    let t = t
        .map_values(&wire_inverse)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;
    let w = SemifiniteFunction(VecArray(
        embedding
            .wires
            .table
            .0
            .iter()
            .map(|wire| host.w.0.0[*wire].clone())
            .collect(),
    ));
    let x = SemifiniteFunction(VecArray(
        embedding
            .operations
            .table
            .0
            .iter()
            .map(|operation| host.x.0.0[*operation].clone())
            .collect(),
    ));

    Ok(Subgraph {
        graph: Hypergraph { s, t, w, x },
        embedding,
    })
}

pub fn glue_embeddings(
    left: &SubgraphEmbedding,
    right: &SubgraphEmbedding,
) -> Result<SubgraphEmbedding, SubgraphError> {
    if left.wires.target != right.wires.target {
        return Err(SubgraphError::WireTargetMismatch);
    }
    if left.operations.target != right.operations.target {
        return Err(SubgraphError::OperationTargetMismatch);
    }

    Ok(SubgraphEmbedding {
        wires: embedding_from_unique_values(
            left.wires
                .table
                .0
                .iter()
                .chain(&right.wires.table.0)
                .copied(),
            left.wires.target,
            |wire| SubgraphError::WireOutOfBounds {
                wire,
                wire_count: left.wires.target,
            },
        )?,
        operations: embedding_from_unique_values(
            left.operations
                .table
                .0
                .iter()
                .chain(&right.operations.table.0)
                .copied(),
            left.operations.target,
            |operation| SubgraphError::OperationOutOfBounds {
                operation,
                operation_count: left.operations.target,
            },
        )?,
    })
}

pub fn glue_subgraphs(
    host: &HostHypergraph,
    left: &SubgraphEmbedding,
    right: &SubgraphEmbedding,
) -> Result<Subgraph, SubgraphError> {
    subgraph_from_embedding(host, glue_embeddings(left, right)?)
}

fn validate_embedding(
    host: &HostHypergraph,
    embedding: &SubgraphEmbedding,
) -> Result<(), SubgraphError> {
    if embedding.wires.target != wire_count(host) {
        return Err(SubgraphError::WireOutOfBounds {
            wire: embedding.wires.target,
            wire_count: wire_count(host),
        });
    }
    if embedding.operations.target != operation_count(host) {
        return Err(SubgraphError::OperationOutOfBounds {
            operation: embedding.operations.target,
            operation_count: operation_count(host),
        });
    }
    if !embedding.wires.is_injective() {
        return Err(SubgraphError::NonInjectiveWireEmbedding);
    }
    if !embedding.operations.is_injective() {
        return Err(SubgraphError::NonInjectiveOperationEmbedding);
    }
    Ok(())
}

fn incident_wires(
    host: &HostHypergraph,
    operations: &FiniteFunction,
) -> Result<FiniteFunction, SubgraphError> {
    let s = host
        .s
        .map_indexes(operations)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;
    let t = host
        .t
        .map_indexes(operations)
        .ok_or(SubgraphError::InvalidIndexedCoproduct)?;
    embedding_from_unique_values(
        s.values.table.0.iter().chain(&t.values.table.0).copied(),
        wire_count(host),
        |wire| SubgraphError::WireOutOfBounds {
            wire,
            wire_count: wire_count(host),
        },
    )
}

fn inverse_on_image(embedding: &FiniteFunction) -> Result<FiniteFunction, SubgraphError> {
    if !embedding.is_injective() {
        return Err(SubgraphError::NonInjectiveWireEmbedding);
    }

    let mut inverse = vec![0; embedding.target];
    for (local, host) in embedding.table.0.iter().copied().enumerate() {
        inverse[host] = local;
    }
    FiniteFunction::new(VecArray(inverse), embedding.table.0.len())
        .ok_or(SubgraphError::InvalidFiniteFunction)
}

fn ensure_incidence_is_in_image(
    incidence: &IndexedCoproduct<FiniteFunction>,
    wires: &FiniteFunction,
) -> Result<(), SubgraphError> {
    for wire in &incidence.values.table.0 {
        if !wires.table.0.contains(wire) {
            return Err(SubgraphError::MissingIncidentWire { wire: *wire });
        }
    }
    Ok(())
}

fn embedding_from_unique_values(
    values: impl IntoIterator<Item = usize>,
    target: usize,
    out_of_bounds: impl Fn(usize) -> SubgraphError,
) -> Result<FiniteFunction, SubgraphError> {
    let mut seen = HashSet::new();
    let mut table = Vec::new();
    for value in values {
        if value >= target {
            return Err(out_of_bounds(value));
        }
        if seen.insert(value) {
            table.push(value);
        }
    }
    FiniteFunction::new(VecArray(table), target).ok_or(SubgraphError::InvalidFiniteFunction)
}

fn wire_count(host: &HostHypergraph) -> usize {
    host.w.0.len()
}

fn operation_count(host: &HostHypergraph) -> usize {
    host.x.0.len()
}
