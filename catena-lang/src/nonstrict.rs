//! Adapters between strict tensor interfaces and explicit product/unit objects.
//!
//! These maps are the packers and unpackers used when interpreting string diagrams for
//! non-strict monoidal categories in a strict hypergraph representation.

use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::{
    category::Arrow,
    lax::{Hyperedge, Hypergraph, OpenHypergraph},
};

const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const PRODUCT_INTRO: &str = "*.intro";
const PRODUCT_ELIM: &str = "*.elim";
const UNIT_INTRO: &str = "unit.intro";
const UNIT_ELIM: &str = "unit.elim";

pub(crate) type Obj = Tree<(), Operation>;
pub(crate) type Arr = Operation;
pub(crate) type Term = OpenHypergraph<Obj, Arr>;

pub(crate) fn to_packer(objects: Vec<Obj>) -> Term {
    objects
        .iter()
        .map(pack_object)
        .fold(OpenHypergraph::empty(), |packed, object| {
            packed.tensor(&object)
        })
}

pub(crate) fn to_unpacker(objects: Vec<Obj>) -> Term {
    objects
        .iter()
        .map(unpack_object)
        .fold(OpenHypergraph::empty(), |unpacked, object| {
            unpacked.tensor(&object)
        })
}

fn pack_object(object: &Obj) -> Term {
    match object {
        Tree::Node(operation, _, children)
            if operation.as_str() == UNIT_TYPE && children.is_empty() =>
        {
            OpenHypergraph::singleton(op(UNIT_INTRO), vec![], vec![object.clone()])
        }
        Tree::Node(operation, _, children) if operation.as_str() == PRODUCT_TYPE => {
            let [left, right] = children.as_slice() else {
                panic!("product object should have exactly two children");
            };
            let children = pack_object(left).tensor(&pack_object(right));
            let intro = OpenHypergraph::singleton(
                op(PRODUCT_INTRO),
                vec![left.clone(), right.clone()],
                vec![object.clone()],
            );
            children
                .compose(&intro)
                .expect("nested product packer should compose")
        }
        _ => OpenHypergraph::identity(vec![object.clone()]),
    }
}

fn unpack_object(object: &Obj) -> Term {
    dual(pack_object(object).map_edges(opposite_operation))
}

fn opposite_operation(operation: Operation) -> Operation {
    match operation.as_str() {
        PRODUCT_INTRO => op(PRODUCT_ELIM),
        UNIT_INTRO => op(UNIT_ELIM),
        _ => panic!("packer contains non-dualizable operation `{operation}`"),
    }
}

fn dual<O, A>(term: OpenHypergraph<O, A>) -> OpenHypergraph<O, A> {
    let OpenHypergraph {
        sources,
        targets,
        hypergraph:
            Hypergraph {
                nodes,
                edges,
                adjacency,
                quotient,
            },
    } = term;

    OpenHypergraph {
        sources: targets,
        targets: sources,
        hypergraph: Hypergraph {
            nodes,
            edges,
            adjacency: adjacency
                .into_iter()
                .map(|Hyperedge { sources, targets }| Hyperedge {
                    sources: targets,
                    targets: sources,
                })
                .collect(),
            quotient,
        },
    }
}

pub(crate) fn unpack_packed_object(object: &Obj, arity: usize) -> Vec<Obj> {
    match arity {
        0 => {
            assert!(
                matches!(
                    object,
                    Tree::Node(operation, _, children)
                        if operation.as_str() == UNIT_TYPE && children.is_empty()
                ),
                "nullary packed object should be unit"
            );
            vec![]
        }
        1 => vec![object.clone()],
        _ => {
            let Tree::Node(operation, _, children) = object else {
                panic!("multi-argument packed object should be product-typed");
            };
            assert_eq!(
                operation.as_str(),
                PRODUCT_TYPE,
                "multi-argument packed object should be product-typed"
            );
            let [left, right] = children.as_slice() else {
                panic!("product object should have exactly two children");
            };
            let mut unpacked = unpack_packed_object(left, arity - 1);
            unpacked.push(right.clone());
            unpacked
        }
    }
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(name: &str) -> Obj {
        Tree::Node(op(name), 0, vec![])
    }

    fn product(left: Obj, right: Obj) -> Obj {
        Tree::Node(op(PRODUCT_TYPE), 0, vec![left, right])
    }

    #[test]
    fn packer_and_unpacker_preserve_top_level_operation_arity() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let ab = product(a.clone(), b.clone());

        let packer = to_packer(vec![ab.clone(), c.clone()]);
        assert_eq!(packer.hypergraph.edges, vec![op(PRODUCT_INTRO)]);
        assert_eq!(
            packer
                .sources
                .iter()
                .map(|node| packer.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![a.clone(), b.clone(), c.clone()]
        );
        assert_eq!(
            packer
                .targets
                .iter()
                .map(|node| packer.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab.clone(), c.clone()]
        );

        let unpacker = to_unpacker(vec![ab.clone(), c.clone()]);
        assert_eq!(unpacker.hypergraph.edges, vec![op(PRODUCT_ELIM)]);
        assert_eq!(
            unpacker
                .sources
                .iter()
                .map(|node| unpacker.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab, c]
        );
        assert_eq!(
            unpacker
                .targets
                .iter()
                .map(|node| unpacker.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![a, b, object("C")]
        );
    }

    #[test]
    fn original_arity_disambiguates_nested_product_arguments() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let ab = product(a.clone(), b.clone());
        let packed = product(ab.clone(), c.clone());

        assert_eq!(
            unpack_packed_object(&packed, 2),
            vec![ab.clone(), c.clone()]
        );
        assert_eq!(unpack_packed_object(&packed, 3), vec![a, b, c]);
    }
}
