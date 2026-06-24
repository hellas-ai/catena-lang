use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};
use std::fmt;

use crate::{
    pass::{PassError, unpack_products::UnpackProductOperation},
    report::TheoryTermMap,
};

pub type Obj = Tree<(), Operation>;

const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationWithBoundarySizes<A> {
    pub operation: A,
    pub source_sizes: Vec<usize>,
    pub target_sizes: Vec<usize>,
}

impl<A: fmt::Display> fmt::Display for OperationWithBoundarySizes<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {:?} -> {:?}",
            self.operation, self.source_sizes, self.target_sizes
        )
    }
}

impl<A: UnpackProductOperation> UnpackProductOperation for OperationWithBoundarySizes<A> {
    fn is_unpack_product_operation(&self) -> bool {
        self.operation.is_unpack_product_operation()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RecordBoundarySizes;

impl<A: Clone> Functor<Obj, A, Obj, OperationWithBoundarySizes<A>> for RecordBoundarySizes {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &A,
        source: &[Obj],
        target: &[Obj],
    ) -> OpenHypergraph<Obj, OperationWithBoundarySizes<A>> {
        OpenHypergraph::singleton(
            OperationWithBoundarySizes {
                operation: a.clone(),
                source_sizes: source.iter().map(object_size).collect(),
                target_sizes: target.iter().map(object_size).collect(),
            },
            source.to_vec(),
            target.to_vec(),
        )
    }

    fn map_arrow(
        &self,
        f: &OpenHypergraph<Obj, A>,
    ) -> OpenHypergraph<Obj, OperationWithBoundarySizes<A>> {
        try_define_map_arrow(self, f)
            .expect("programmer error: record-boundary-sizes is not a functor")
    }
}

pub fn run<A: Clone>(
    terms: &TheoryTermMap<A>,
) -> Result<TheoryTermMap<OperationWithBoundarySizes<A>>, PassError> {
    terms
        .iter()
        .map(|(theory_id, definitions)| {
            let definitions = definitions
                .iter()
                .map(|(definition_name, term)| {
                    let mut transformed = RecordBoundarySizes.map_arrow(term);
                    transformed.quotient().map_err(|_| PassError::Quotient {
                        pass: "record_boundary_sizes",
                        theory: theory_id.to_string(),
                        definition: definition_name.to_string(),
                    })?;
                    Ok((definition_name.clone(), transformed))
                })
                .collect::<Result<_, PassError>>()?;
            Ok((theory_id.clone(), definitions))
        })
        .collect()
}

pub fn object_size(o: &Obj) -> usize {
    match o {
        Tree::Empty => 0,
        Tree::Node(op, _, children) if op.as_str() == UNIT_TYPE && children.is_empty() => 0,
        Tree::Node(op, _, children) if op.as_str() == PRODUCT_TYPE => {
            children.iter().map(object_size).sum()
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }

    fn ty(name: &str) -> Obj {
        Tree::Node(op(name), 0, vec![])
    }

    fn product(children: Vec<Obj>) -> Obj {
        Tree::Node(op(PRODUCT_TYPE), 0, children)
    }

    #[test]
    fn records_source_and_target_object_sizes() {
        let source = vec![product(vec![ty("A"), ty("B")]), ty("C")];
        let target = vec![ty("D")];

        let mapped = RecordBoundarySizes.map_operation(&op("f0"), &source, &target);
        let label = &mapped.hypergraph.edges[0];

        assert_eq!(label.operation, op("f0"));
        assert_eq!(label.source_sizes, vec![2, 1]);
        assert_eq!(label.target_sizes, vec![1]);
        assert_eq!(mapped.hypergraph.nodes[0], source[0]);
        assert_eq!(mapped.hypergraph.nodes[1], source[1]);
        assert_eq!(mapped.hypergraph.nodes[2], target[0]);
    }

    #[test]
    fn records_sizes_without_requiring_operation_labels() {
        let source = vec![product(vec![ty("A"), ty("B")])];
        let target = vec![ty("C")];

        let mapped = RecordBoundarySizes.map_operation(&"f0".to_string(), &source, &target);
        let label = &mapped.hypergraph.edges[0];

        assert_eq!(label.operation, "f0");
        assert_eq!(label.source_sizes, vec![2]);
        assert_eq!(label.target_sizes, vec![1]);
    }

    #[test]
    fn records_zero_size_unit_components() {
        let source = vec![ty("A"), ty(UNIT_TYPE), ty("B")];
        let target = vec![ty("C")];

        let mapped = RecordBoundarySizes.map_operation(&op("f0"), &source, &target);
        let label = &mapped.hypergraph.edges[0];

        assert_eq!(label.source_sizes, vec![1, 0, 1]);
        assert_eq!(
            mapped.hypergraph.nodes,
            vec![ty("A"), ty(UNIT_TYPE), ty("B"), ty("C")]
        );
    }

    #[test]
    fn unit_and_empty_have_size_zero() {
        assert_eq!(object_size(&Tree::Empty), 0);
        assert_eq!(object_size(&ty(UNIT_TYPE)), 0);
        assert_eq!(object_size(&product(vec![ty("A"), ty(UNIT_TYPE)])), 1);
    }
}
