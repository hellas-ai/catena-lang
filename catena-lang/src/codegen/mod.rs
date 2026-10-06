//! Code generator selection and dispatch.
//!
//! Each backend owns its lowering, intermediate representation, and rendering.
//! The public entrypoint returns a runtime module; backend IR stays internal.

mod default;
mod dialect;
mod experimental;

use hexpr::Operation;
use thiserror::Error;

use crate::{
    pass::record_boundary_sizes::OperationWithBoundarySizes, report::TheoryTermMap,
    runtime::RuntimeModule,
};

pub use crate::runtime::GpuDialect;

type CodegenTermMap = TheoryTermMap<OperationWithBoundarySizes<Operation>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CodegenKind {
    #[default]
    Default,
    Experimental,
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Default(#[from] default::CodegenError),
    #[error(transparent)]
    Experimental(#[from] experimental::CodegenError),
}

/// Generate C++ source and runtime metadata from closure-converted terms.
pub fn codegen(
    kind: CodegenKind,
    terms: &CodegenTermMap,
    dialect: GpuDialect,
) -> Result<RuntimeModule, CodegenError> {
    assert_closures_converted(terms);
    match kind {
        CodegenKind::Default => default::codegen(terms, dialect).map_err(CodegenError::from),
        CodegenKind::Experimental => {
            experimental::codegen(terms, dialect).map_err(CodegenError::from)
        }
    }
}

/// Assert that closure conversion has eliminated all closure operations and types.
fn assert_closures_converted(terms: &CodegenTermMap) {
    for definitions in terms.values() {
        for term in definitions.values() {
            for edge in &term.hypergraph.edges {
                let op = edge.operation.as_str();
                assert!(
                    !matches!(
                        op,
                        "defer"
                            | "run"
                            | "compose"
                            | "tensor"
                            | "lift"
                            | "evaluate"
                            | "closure.marker"
                    ),
                    "unconverted closure operation `{op}` reached codegen: closures must have been converted"
                );
            }
            let mut pending: Vec<_> = term.hypergraph.nodes.iter().collect();
            while let Some(ty) = pending.pop() {
                if let metacat::tree::Tree::Node(op, _, args) = ty {
                    assert!(
                        op.as_str() != "=>",
                        "unconverted closure type reached codegen: {ty:?}; closures must have been converted"
                    );
                    pending.extend(args);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metacat::{theory::TheoryId, tree::Tree};
    use open_hypergraphs::lax::OpenHypergraph;

    #[test]
    fn both_backends_reject_unconverted_closures_at_entry() {
        for kind in [CodegenKind::Default, CodegenKind::Experimental] {
            for (name, output) in [
                ("defer", Tree::Empty),
                ("run", Tree::Empty),
                ("compose", Tree::Empty),
                ("tensor", Tree::Empty),
                ("lift", Tree::Empty),
                ("evaluate", Tree::Empty),
                ("closure.marker", Tree::Empty),
                (
                    "opaque",
                    Tree::Node(
                        "*".parse().unwrap(),
                        0,
                        vec![
                            Tree::Empty,
                            Tree::Node("=>".parse().unwrap(), 0, vec![Tree::Empty, Tree::Empty]),
                        ],
                    ),
                ),
                (
                    "opaque",
                    Tree::Node("=>".parse().unwrap(), 0, vec![Tree::Empty, Tree::Empty]),
                ),
            ] {
                let term = OpenHypergraph::singleton(
                    OperationWithBoundarySizes {
                        operation: name.parse().unwrap(),
                        source_sizes: vec![],
                        target_sizes: vec![1],
                    },
                    vec![],
                    vec![output],
                );
                let terms = std::collections::BTreeMap::from([(
                    TheoryId("program".parse().unwrap()),
                    std::collections::BTreeMap::from([("test".parse().unwrap(), term)]),
                )]);
                let panic = std::panic::catch_unwind(|| codegen(kind, &terms, GpuDialect::Cuda))
                    .expect_err("unconverted closures must panic at the codegen boundary");
                assert!(
                    panic
                        .downcast_ref::<String>()
                        .unwrap()
                        .contains("closures must have been converted")
                );
            }
        }
    }
}
