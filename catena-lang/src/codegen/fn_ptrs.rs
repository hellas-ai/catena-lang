use std::collections::HashMap;

use hexpr::Operation;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{check::AnnotatedTerm, pass::record_boundary_sizes::OperationWithBoundarySizes};

const NAME_PREFIX: &str = "name.";

pub type FnPtrNodeMap = HashMap<NodeId, FnPtrSymbol>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FnPtrSymbol {
    pub target: Operation,
}

#[derive(Debug, Error)]
pub enum FnPtrSymbolError {
    #[error(
        "function pointer symbol `{operation}` should produce exactly one target, found {target_count}"
    )]
    InvalidTargetCount {
        operation: Operation,
        target_count: usize,
    },
    #[error(
        "function pointer symbol `{operation}` should not overwrite an existing symbol on node {node}"
    )]
    DuplicateNodeSymbol { operation: Operation, node: usize },
    #[error("generated function pointer target `{0}` is not a valid operation")]
    InvalidTargetOperation(String),
}

/// Find direct `name.*` function pointer symbols produced inside a definition.
///
/// The returned node ids are ids in the quotiented graph, matching the graph shape used by
/// later SSA/codegen passes. This is intentionally a partial map: externally supplied function
/// pointer wires, values flowing through ordinary operations, and conditionally-produced function
/// pointers are not resolved here.
pub fn direct_fn_ptr_symbols(
    term: &AnnotatedTerm<OperationWithBoundarySizes<Operation>>,
) -> Result<FnPtrNodeMap, FnPtrSymbolError> {
    let mut term = term.clone();
    term.quotient().ok();

    let mut symbols = HashMap::new();

    for (edge_index, operation) in term.hypergraph.edges.iter().enumerate() {
        let Some(target_name) = operation.operation.as_str().strip_prefix(NAME_PREFIX) else {
            continue;
        };

        let adjacency = &term.hypergraph.adjacency[edge_index];
        let [target] = adjacency.targets.as_slice() else {
            return Err(FnPtrSymbolError::InvalidTargetCount {
                operation: operation.operation.clone(),
                target_count: adjacency.targets.len(),
            });
        };
        if symbols.contains_key(target) {
            return Err(FnPtrSymbolError::DuplicateNodeSymbol {
                operation: operation.operation.clone(),
                node: target.0,
            });
        }

        symbols.insert(
            *target,
            FnPtrSymbol {
                target: target_name.parse().map_err(|_| {
                    FnPtrSymbolError::InvalidTargetOperation(target_name.to_string())
                })?,
            },
        );
    }

    Ok(symbols)
}

#[cfg(test)]
mod tests {
    use super::*;

    use metacat::tree::Tree;
    use open_hypergraphs::lax::OpenHypergraph;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn sized_op(name: &str) -> OperationWithBoundarySizes<Operation> {
        OperationWithBoundarySizes {
            operation: op(name),
            source_sizes: Vec::new(),
            target_sizes: Vec::new(),
        }
    }

    fn ty(name: &str) -> Tree<(), Operation> {
        Tree::Node(op(name), 0, vec![])
    }

    #[test]
    fn maps_name_operation_target_node_to_symbol() {
        let term = OpenHypergraph::singleton(sized_op("name.bool.not"), vec![], vec![ty("->")]);

        let symbols = direct_fn_ptr_symbols(&term).unwrap();

        assert_eq!(
            symbols.get(&NodeId(0)),
            Some(&FnPtrSymbol {
                target: op("bool.not"),
            })
        );
    }

    #[test]
    fn ignores_non_name_operations() {
        let term =
            OpenHypergraph::singleton(sized_op("bool.not"), vec![ty("bool")], vec![ty("bool")]);

        assert!(direct_fn_ptr_symbols(&term).unwrap().is_empty());
    }
}
