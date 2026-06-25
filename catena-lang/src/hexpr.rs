use hexpr::{Hexpr, Operation, Variable};
use metacat::tree::Tree;
use open_hypergraphs::lax::OpenHypergraph;

type Obj = Tree<(), Operation>;

pub fn objects_to_hexpr(objects: &[Obj]) -> Hexpr {
    match objects {
        [] => Hexpr::Operation(op("1")),
        [object] => object_to_hexpr(object),
        _ => Hexpr::Tensor(objects.iter().map(object_to_hexpr).collect()),
    }
}

pub fn term_to_hexpr(term: &OpenHypergraph<Obj, Operation>) -> Hexpr {
    let node_vars = (0..term.hypergraph.nodes.len())
        .map(|index| var(&format!("w{index}")))
        .collect::<Vec<_>>();
    let mut parts = Vec::new();

    parts.push(Hexpr::Frobenius {
        sources: vars_for(&node_vars, &term.sources),
        targets: vec![],
    });

    for (edge, hyperedge) in term.hypergraph.edges.iter().zip(&term.hypergraph.adjacency) {
        parts.push(Hexpr::Composition(vec![
            Hexpr::Frobenius {
                sources: vec![],
                targets: vars_for(&node_vars, &hyperedge.sources),
            },
            Hexpr::Operation(edge.clone()),
            Hexpr::Frobenius {
                sources: vars_for(&node_vars, &hyperedge.targets),
                targets: vec![],
            },
        ]));
    }

    parts.push(Hexpr::Frobenius {
        sources: vec![],
        targets: vars_for(&node_vars, &term.targets),
    });

    Hexpr::Composition(parts)
}

fn object_to_hexpr(object: &Obj) -> Hexpr {
    match object {
        Tree::Empty => Hexpr::Operation(op("1")),
        Tree::Leaf(index, _) => {
            let variable = var(&format!("p{index}"));
            Hexpr::Frobenius {
                sources: vec![variable.clone()],
                targets: vec![variable],
            }
        }
        Tree::Node(operation, _, children) => match children.as_slice() {
            [] => Hexpr::Operation(operation.clone()),
            _ => Hexpr::Composition(vec![
                Hexpr::Tensor(children.iter().map(object_to_hexpr).collect()),
                Hexpr::Operation(operation.clone()),
            ]),
        },
    }
}

fn vars_for(node_vars: &[Variable], nodes: &[open_hypergraphs::lax::NodeId]) -> Vec<Variable> {
    nodes.iter().map(|node| node_vars[node.0].clone()).collect()
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

fn var(name: &str) -> Variable {
    name.parse().expect("generated variable should parse")
}

#[cfg(test)]
mod tests {
    use hexpr::{Signature, try_interpret};
    use open_hypergraphs::lax::OpenHypergraph;

    use super::*;

    struct TestSignature;

    impl Signature for TestSignature {
        type Arr = Operation;
        type Obj = ();
        type Error = String;

        fn try_parse_op(&self, op: &Operation) -> Result<Self::Arr, Self::Error> {
            Ok(op.clone())
        }

        fn profile(&self, op: &Self::Arr) -> (Vec<Option<Self::Obj>>, Vec<Option<Self::Obj>>) {
            match op.to_string().as_str() {
                "f" | "g" | "val" => (vec![Some(())], vec![Some(())]),
                "split" => (vec![Some(())], vec![Some(()), Some(())]),
                "merge" => (vec![Some(()), Some(())], vec![Some(())]),
                "bool" | "f32" | "1" => (vec![], vec![Some(())]),
                other => panic!("unexpected test operation `{other}`"),
            }
        }
    }

    #[test]
    fn term_to_hexpr_roundtrips_graph_shape_up_to_variable_names() {
        let parsed = parse("([x] f [y . y y] {g g} merge)");
        let mut term = try_interpret(&TestSignature, &parsed)
            .expect("test hexpr should interpret")
            .map_nodes(|_| Tree::Empty);
        term.quotient().expect("test hexpr should quotient");

        let roundtripped = term_to_hexpr(&term);
        println!("{}", roundtripped);

        assert_eq!(normalized(&roundtripped), normalized(&parsed));
    }

    #[test]
    fn objects_to_hexpr_roundtrips_object_map_up_to_variable_names() {
        let expected = parse("{(bool val) (f32 val)}");
        let generated = objects_to_hexpr(&[
            object("val", vec![object("bool", vec![])]),
            object("val", vec![object("f32", vec![])]),
        ]);

        assert_eq!(normalized(&generated), normalized(&expected));
    }

    fn normalized(hexpr: &Hexpr) -> OpenHypergraph<(), Operation> {
        let mut term = try_interpret(&TestSignature, hexpr)
            .expect("test hexpr should interpret")
            .map_nodes(|_| ());
        term.quotient().expect("test hexpr should quotient");
        term
    }

    fn parse(source: &str) -> Hexpr {
        source.parse().expect("test hexpr should parse")
    }

    fn object(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }
}
