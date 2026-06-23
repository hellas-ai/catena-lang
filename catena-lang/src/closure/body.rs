use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

use crate::check::AnnotatedTerm;

const CLOSURE_TYPE: &str = "=>";
const UNIT_TYPE: &str = "1";
const DEFER: &str = "defer";
const COMPOSE: &str = "compose";
const RUN: &str = "run";

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

    body.sources.push(argument);
    body.targets = vec![output];

    Ok(body)
}

fn closure_parts(object: &Obj) -> Option<(&Obj, &Obj)> {
    let Tree::Node(operation, _, children) = object else {
        return None;
    };
    if operation.as_str() != CLOSURE_TYPE {
        return None;
    }
    let [domain, codomain] = children.as_slice() else {
        return None;
    };
    Some((domain, codomain))
}

fn closure_type_of(domain: Obj, codomain: Obj) -> Obj {
    Tree::Node(op(CLOSURE_TYPE), 0, vec![domain, codomain])
}

fn unit_type() -> Obj {
    Tree::Node(op(UNIT_TYPE), 0, vec![])
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

#[cfg(test)]
mod tests {
    use metacat::{
        theory::{RawTheorySet, Theory, TheoryId, TheorySet},
        tree::Tree,
    };
    use open_hypergraphs::lax::NodeId;

    use super::*;
    use crate::{
        check::{DefinitionTypes, check},
        closure::{extract::extract_region, region::closure_region},
        elaborate::elaborate,
        stdlib,
    };

    #[test]
    fn closure_body_has_environment_and_argument_sources() {
        let definition = annotated_program_definition(
            r#"
            (def program run-bool-id : (bool val) -> ({1 (bool val)} =>) = (
              {[x] bool.t}
              bool.and
              bool.not
              {defer (name.bool.id lift)}
              compose
            ))
            "#,
            "run-bool-id",
        );
        let [region] = closure_region(&definition, &[definition.targets[0]])
            .expect("region discovery should succeed")
            .try_into()
            .expect("expected one closure region");
        let extracted =
            extract_region(&definition, &region).expect("region extraction should succeed");
        let body = closure_body(&extracted).expect("closure body construction should succeed");

        assert_eq!(
            body.hypergraph.edges.len(),
            extracted.hypergraph.edges.len() + 3
        );
        assert_eq!(
            interface_types(&body, &body.sources),
            vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![]),]
        );
        assert_eq!(
            interface_types(&body, &body.targets),
            vec![obj("val", vec![obj("bool", vec![])])]
        );
    }

    fn theories_with(source: &'static str) -> (TheorySet, DefinitionTypes) {
        let raw_theories = RawTheorySet::from_texts(stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = elaborate(raw_theories).expect("test theories should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theories should load");
        let definition_types = check(&theory_set).expect("test theories should typecheck");
        (theory_set, definition_types)
    }

    fn annotated_program_definition(source: &'static str, definition: &str) -> AnnotatedTerm {
        let (theory_set, definition_types) = theories_with(source);
        let program = TheoryId("program".parse().expect("program theory id should parse"));
        let definition: Operation = definition
            .parse()
            .expect("program definition name should parse");
        let theory = theory_set
            .theories
            .get(&program)
            .expect("program theory should exist");
        let Theory::Theory { arrows, .. } = theory else {
            panic!("program should be a theory");
        };
        let arrow = arrows
            .get(&definition)
            .expect("program definition should exist");
        let mut body = arrow
            .definition
            .clone()
            .expect("program arrow should be a definition");
        body.quotient().ok();
        let labels = definition_types
            .get(&program)
            .and_then(|definitions| definitions.get(&definition))
            .cloned()
            .expect("program definition should have checked node types");
        body.with_nodes(|_| labels)
            .expect("checked node labels should match definition graph")
    }

    fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
        interface
            .iter()
            .map(|node| term.hypergraph.nodes[node.0].clone())
            .collect()
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }
}
