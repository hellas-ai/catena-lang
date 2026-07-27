//! Adapters between strict tensor interfaces and explicit product/unit objects.
//!
//! Flatteners/unflatteners adapt one object to or from its strict, flattened
//! tensor interface. Packers/unpackers adapt a tensor interface to or from one
//! right-associated packed product object.
//!
//! For theoretical background, see
//! [String Diagrams for Strictification and Coherence](https://arxiv.org/abs/2201.11738)

use hexpr::{Hexpr, Operation, Variable};
use metacat::tree::Tree;
use open_hypergraphs::{
    category::Arrow,
    lax::{Hyperedge, Hypergraph, NodeId, OpenHypergraph},
};

use crate::stdlib::constants::{
    PRODUCT_ELIM, PRODUCT_INTRO, PRODUCT_TYPE, UNIT_ELIM, UNIT_INTRO, UNIT_TYPE,
};

pub(crate) type Obj = Tree<(), Operation>;
pub(crate) type Arr = Operation;
pub(crate) type Term = OpenHypergraph<Obj, Arr>;

/// Build the type map from a strict tensor boundary to its canonical packed
/// object. Packing is right-associated, as it is for [`to_packer`].
pub(crate) fn packed_type_map<E>(
    object_count: usize,
    fresh_variable: &mut impl FnMut() -> Result<Variable, E>,
) -> Result<Hexpr, E> {
    match object_count {
        0 => Ok(Hexpr::Operation(op(UNIT_TYPE))),
        1 => Ok(identity_hexpr(fresh_variable()?)),
        2 => Ok(Hexpr::Operation(op(PRODUCT_TYPE))),
        n => Ok(Hexpr::Composition(vec![
            Hexpr::Tensor(vec![
                identity_hexpr(fresh_variable()?),
                packed_type_map(n - 1, fresh_variable)?,
            ]),
            Hexpr::Operation(op(PRODUCT_TYPE)),
        ])),
    }
}

/// Add operations which pack `nodes` into one right-associated output node.
///
/// This is shared by type maps (`1` and `*`) and program terms
/// (`unit.intro` and `*.intro`) so both representations use the same shape.
pub(crate) fn pack_nodes<O: Clone>(
    term: &mut OpenHypergraph<O, Operation>,
    nodes: &[NodeId],
    node_label: O,
    unit_operation: &str,
    product_operation: &str,
) -> NodeId {
    match nodes {
        [] => {
            let packed = term.new_node(node_label);
            term.new_edge(op(unit_operation), (vec![], vec![packed]));
            packed
        }
        [only] => *only,
        [head, tail @ ..] => {
            let packed_tail = pack_nodes(
                term,
                tail,
                node_label.clone(),
                unit_operation,
                product_operation,
            );
            let packed = term.new_node(node_label);
            term.new_edge(
                op(product_operation),
                (vec![*head, packed_tail], vec![packed]),
            );
            packed
        }
    }
}

/// Compute the size of an object: equivalent to `size` in
/// [String Diagrams for Strictification and Coherence](https://arxiv.org/abs/2201.11738)
pub(crate) fn object_size(object: &Obj) -> usize {
    match object {
        Tree::Empty => 0,
        Tree::Node(operation, _, children)
            if operation.as_str() == UNIT_TYPE && children.is_empty() =>
        {
            0
        }
        Tree::Node(operation, _, children) if operation.as_str() == PRODUCT_TYPE => {
            children.iter().map(object_size).sum()
        }
        _ => 1,
    }
}

pub(crate) fn to_unflattener(object: &Obj) -> Term {
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
            let children = to_unflattener(left).tensor(&to_unflattener(right));
            let intro = OpenHypergraph::singleton(
                op(PRODUCT_INTRO),
                vec![left.clone(), right.clone()],
                vec![object.clone()],
            );
            children
                .compose(&intro)
                .expect("nested product unflattener should compose")
        }
        _ => OpenHypergraph::identity(vec![object.clone()]),
    }
}

pub(crate) fn to_flattener(object: &Obj) -> Term {
    dual(to_unflattener(object).map_edges(opposite_operation))
}

pub(crate) fn to_unflatteners(objects: &[Obj]) -> Term {
    objects
        .iter()
        .map(to_unflattener)
        .fold(OpenHypergraph::empty(), |unflattened, object| {
            unflattened.tensor(&object)
        })
}

pub(crate) fn to_flatteners(objects: &[Obj]) -> Term {
    objects
        .iter()
        .map(to_flattener)
        .fold(OpenHypergraph::empty(), |flattened, object| {
            flattened.tensor(&object)
        })
}

pub(crate) fn to_packer(objects: Vec<Obj>) -> Term {
    packer_for(&objects)
}

pub(crate) fn to_unpacker(objects: Vec<Obj>) -> Term {
    dual(to_packer(objects).map_edges(opposite_operation))
}

fn packer_for(objects: &[Obj]) -> Term {
    let packed = pack_objects(objects);
    match objects {
        [] => OpenHypergraph::singleton(op(UNIT_INTRO), vec![], vec![packed]),
        [only] => OpenHypergraph::identity(vec![only.clone()]),
        [head, tail @ ..] => {
            let packed_tail = pack_objects(tail);
            let intro = OpenHypergraph::singleton(
                op(PRODUCT_INTRO),
                vec![head.clone(), packed_tail],
                vec![packed],
            );
            OpenHypergraph::identity(vec![head.clone()])
                .tensor(&packer_for(tail))
                .compose(&intro)
                .expect("product packer should compose")
        }
    }
}

pub(crate) fn pack_objects(objects: &[Obj]) -> Obj {
    match objects {
        [] => Tree::Node(op(UNIT_TYPE), 0, vec![]),
        [only] => only.clone(),
        [head, tail @ ..] => {
            Tree::Node(op(PRODUCT_TYPE), 0, vec![head.clone(), pack_objects(tail)])
        }
    }
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
            let mut unpacked = vec![left.clone()];
            unpacked.extend(unpack_packed_object(right, arity - 1));
            unpacked
        }
    }
}

fn identity_hexpr(variable: Variable) -> Hexpr {
    Hexpr::Frobenius {
        sources: vec![variable.clone()],
        targets: vec![variable],
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
    fn type_map_packing_uses_the_canonical_right_association() {
        let mut next = 0;
        let packed = packed_type_map(3, &mut || {
            let variable = format!("p{next}").parse::<Variable>().unwrap();
            next += 1;
            Ok::<_, ()>(variable)
        })
        .unwrap();

        let Hexpr::Composition(parts) = packed else {
            panic!("three objects should require a composite type map");
        };
        let [Hexpr::Tensor(children), Hexpr::Operation(product)] = parts.as_slice() else {
            panic!("three objects should pack as A * (B * C)");
        };
        assert!(matches!(children[0], Hexpr::Frobenius { .. }));
        assert!(
            matches!(&children[1], Hexpr::Operation(operation) if operation.as_str() == PRODUCT_TYPE)
        );
        assert_eq!(product.as_str(), PRODUCT_TYPE);
    }

    #[test]
    fn flatteners_adapt_one_object_to_its_flattened_components() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let ab = product(a.clone(), b.clone());

        let unflattener = to_unflatteners(&[ab.clone(), c.clone()]);
        assert_eq!(unflattener.hypergraph.edges, vec![op(PRODUCT_INTRO)]);
        assert_eq!(
            unflattener
                .sources
                .iter()
                .map(|node| unflattener.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![a.clone(), b.clone(), c.clone()]
        );
        assert_eq!(
            unflattener
                .targets
                .iter()
                .map(|node| unflattener.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab.clone(), c.clone()]
        );

        let flattener = to_flatteners(&[ab.clone(), c.clone()]);
        assert_eq!(flattener.hypergraph.edges, vec![op(PRODUCT_ELIM)]);
        assert_eq!(
            flattener
                .sources
                .iter()
                .map(|node| flattener.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab, c]
        );
        assert_eq!(
            flattener
                .targets
                .iter()
                .map(|node| flattener.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![a, b, object("C")]
        );
    }

    #[test]
    fn packers_adapt_an_interface_to_one_right_associated_product() {
        let width = object("width");
        let buffer = object("buffer");
        let row = object("row");
        let col = object("col");
        let e = object("E");
        let ab = product(width, buffer);
        let cd = product(row, col);
        let packed = product(ab.clone(), product(cd.clone(), e.clone()));

        let packer = to_packer(vec![ab.clone(), cd.clone(), e.clone()]);
        assert_eq!(
            packer.hypergraph.edges,
            vec![op(PRODUCT_INTRO), op(PRODUCT_INTRO)]
        );
        assert_eq!(
            packer
                .sources
                .iter()
                .map(|node| packer.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab.clone(), cd.clone(), e.clone()]
        );
        assert_eq!(
            packer
                .targets
                .iter()
                .map(|node| packer.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![packed.clone()]
        );

        let unpacker = to_unpacker(vec![ab.clone(), cd.clone(), e.clone()]);
        assert_eq!(
            unpacker.hypergraph.edges,
            vec![op(PRODUCT_ELIM), op(PRODUCT_ELIM)]
        );
        assert_eq!(
            unpacker
                .sources
                .iter()
                .map(|node| unpacker.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![packed]
        );
        assert_eq!(
            unpacker
                .targets
                .iter()
                .map(|node| unpacker.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![ab, cd, e]
        );
    }

    #[test]
    fn original_arity_disambiguates_nested_product_arguments() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let bc = product(b.clone(), c.clone());
        let packed = product(a.clone(), bc.clone());

        assert_eq!(
            unpack_packed_object(&packed, 2),
            vec![a.clone(), bc.clone()]
        );
        assert_eq!(unpack_packed_object(&packed, 3), vec![a, b, c]);
    }

    #[test]
    fn object_size_counts_flattened_product_components() {
        assert_eq!(object_size(&Tree::Empty), 0);
        assert_eq!(object_size(&object(UNIT_TYPE)), 0);
        assert_eq!(
            object_size(&product(
                object("A"),
                product(object(UNIT_TYPE), object("B"))
            )),
            2
        );
    }
}
