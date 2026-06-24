use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

pub type Obj = Tree<(), Operation>;
pub type Arr = Operation;

const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OperationWithSizes {
    pub operation: Operation,
    pub source_sizes: Vec<usize>,
    pub target_sizes: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RecordObjectSizes;

impl Functor<Obj, Arr, Obj, OperationWithSizes> for RecordObjectSizes {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> OpenHypergraph<Obj, OperationWithSizes> {
        OpenHypergraph::singleton(
            OperationWithSizes {
                operation: a.clone(),
                source_sizes: source.iter().map(object_size).collect(),
                target_sizes: target.iter().map(object_size).collect(),
            },
            source.to_vec(),
            target.to_vec(),
        )
    }

    fn map_arrow(&self, f: &OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, OperationWithSizes> {
        try_define_map_arrow(self, f)
            .expect("programmer error: record-object-sizes is not a functor")
    }
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

        let mapped = RecordObjectSizes.map_operation(&op("f0"), &source, &target);
        let label = &mapped.hypergraph.edges[0];

        assert_eq!(label.operation, op("f0"));
        assert_eq!(label.source_sizes, vec![2, 1]);
        assert_eq!(label.target_sizes, vec![1]);
        assert_eq!(mapped.hypergraph.nodes[0], source[0]);
        assert_eq!(mapped.hypergraph.nodes[1], source[1]);
        assert_eq!(mapped.hypergraph.nodes[2], target[0]);
    }

    #[test]
    fn unit_and_empty_have_size_zero() {
        assert_eq!(object_size(&Tree::Empty), 0);
        assert_eq!(object_size(&ty(UNIT_TYPE)), 0);
        assert_eq!(object_size(&product(vec![ty("A"), ty(UNIT_TYPE)])), 1);
    }
}
