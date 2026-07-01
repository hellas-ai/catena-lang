use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    nonstrict::to_unpacker,
    stdlib::constants::{COMPOSE, DEFER, FN_HOM_TYPE, RUN, UNIT_TYPE},
};

type Obj = Tree<(), Operation>;

#[derive(Debug, Error)]
pub enum ClosureBodyError {
    #[error("extracted closure region must have exactly one target, found {actual}")]
    TargetArity { actual: usize },
    #[error("extracted closure region target node n{wire} is out of bounds")]
    TargetOutOfBounds { wire: usize },
    #[error("extracted closure region target node n{wire} is not closure-typed")]
    TargetNotClosureTyped { wire: usize },
}

/// Build the function body for an extracted closure region.
///
/// Given an extracted term `t : X -> (A => B)`, this produces a term
/// `X, A -> B` by appending the inlined evaluation sequence:
/// `defer ; compose ; run`.
pub fn closure_body(extracted: &AnnotatedTerm) -> Result<AnnotatedTerm, ClosureBodyError> {
    let [closure_wire] = extracted.targets.as_slice() else {
        return Err(ClosureBodyError::TargetArity {
            actual: extracted.targets.len(),
        });
    };
    let closure_type = extracted.hypergraph.nodes.get(closure_wire.0).ok_or(
        ClosureBodyError::TargetOutOfBounds {
            wire: closure_wire.0,
        },
    )?;
    let (domain, codomain) =
        closure_parts(closure_type).ok_or(ClosureBodyError::TargetNotClosureTyped {
            wire: closure_wire.0,
        })?;

    let mut body = extracted.clone();
    let unit = unit_type();
    let environment = packed_environment_source(&mut body);

    let argument = body.new_node(domain.clone());
    let deferred_argument = body.new_node(closure_type_of(unit.clone(), domain.clone()));
    let composed = body.new_node(closure_type_of(unit, codomain.clone()));
    let output = body.new_node(codomain.clone());

    body.new_edge(op(DEFER), (vec![argument], vec![deferred_argument]));
    body.new_edge(
        op(COMPOSE),
        (vec![deferred_argument, *closure_wire], vec![composed]),
    );
    body.new_edge(op(RUN), (vec![composed], vec![output]));

    body.sources = vec![environment, argument];
    body.targets = vec![output];

    Ok(body)
}

fn packed_environment_source(body: &mut AnnotatedTerm) -> NodeId {
    let components = body.sources.clone();
    let component_types = interface_types(body, &components);
    let unpacker = to_unpacker(component_types);
    let (sources, targets) = body.append(unpacker);

    let [source] = sources.as_slice() else {
        unreachable!("one packed environment object should produce one source");
    };
    assert_eq!(
        targets.len(),
        components.len(),
        "environment unpacker should reproduce the extracted environment"
    );
    for (target, component) in targets.into_iter().zip(components) {
        body.unify(target, component);
    }
    *source
}

fn closure_parts(object: &Obj) -> Option<(&Obj, &Obj)> {
    let Tree::Node(operation, _, children) = object else {
        return None;
    };
    if operation.as_str() != FN_HOM_TYPE {
        return None;
    }
    let [domain, codomain] = children.as_slice() else {
        return None;
    };
    Some((domain, codomain))
}

fn closure_type_of(domain: Obj, codomain: Obj) -> Obj {
    Tree::Node(op(FN_HOM_TYPE), 0, vec![domain, codomain])
}

fn unit_type() -> Obj {
    Tree::Node(op(UNIT_TYPE), 0, vec![])
}

fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
