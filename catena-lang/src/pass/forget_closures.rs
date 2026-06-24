//! Forget closure operations by wire bending.
use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::category::Arrow;
use open_hypergraphs::lax::{
    NodeId, OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};
use thiserror::Error;

use crate::{
    check::DefinitionTypes,
    nonstrict::{to_packer, to_unpacker, unpack_packed_object},
    report::{AnnotatedTerm, TheoryTermMap},
};

const CLOSURE_TYPE: &str = "=>";
const FUNCTION_TYPE: &str = "->";
const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const VALUE_TYPE: &str = "val";
const NAME_PREFIX: &str = "name.";
const DEFER: &str = "defer";
const RUN: &str = "run";
const COMPOSE: &str = "compose";
const TENSOR: &str = "tensor";
const LIFT: &str = "lift";
const EVAL: &str = "eval";

pub type Obj = Tree<(), Operation>;
pub type Arr = Operation;

#[derive(Debug, Error)]
pub enum ForgetClosuresError {
    #[error("missing definition `{definition}` in theory `{theory}`")]
    MissingDefinition { theory: String, definition: String },
    #[error("missing checked node types for definition `{definition}` in theory `{theory}`")]
    MissingDefinitionTypes { theory: String, definition: String },
    #[error(
        "typechecked node label count mismatch for definition `{definition}` in theory `{theory}`"
    )]
    NodeLabelCountMismatch { theory: String, definition: String },
}

/// Run the "forget closures" pass, removing all closed monoidal category operations from each term
/// in a theory.
pub fn run(
    theory_set: &TheorySet,
    definition_types: &DefinitionTypes,
) -> Result<TheoryTermMap, ForgetClosuresError> {
    let mut output = BTreeMap::new();

    for (theory_id, theory) in &theory_set.theories {
        let Theory::Theory { arrows, .. } = theory else {
            continue;
        };

        let mut transformed = BTreeMap::new();
        let theory_definition_types = definition_types.get(theory_id);

        // Loop through arrows of the theory, applying ForgetClosures functor to each
        for (definition_name, arrow) in arrows {
            let Some(_) = &arrow.definition else {
                continue;
            };

            let typed =
                typed_definition(theory_id, definition_name, theory, theory_definition_types)?;
            let mut transformed_definition = ForgetClosures { theory }.map_arrow(&typed);
            transformed_definition.quotient().ok();
            transformed.insert(definition_name.clone(), transformed_definition);
        }

        if !transformed.is_empty() {
            output.insert(theory_id.clone(), transformed);
        }
    }

    Ok(output)
}

/// A functor removing all closed monoidal category operations (defer, run, compose, tensor) from a
/// term by "bending wires".
/// On objects, we forget all products. So for example, A×B → A●B.
/// This breaks non-CMC operations, which we wrap in adapters.
/// For example, for operation `f`, we get `Φ ; f ; Φ⁻¹`
#[derive(Clone)]
struct ForgetClosures<'a> {
    theory: &'a Theory,
}

impl Functor<Obj, Arr, Obj, Arr> for ForgetClosures<'_> {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        expand_object(o).into_iter()
    }

    fn map_operation(&self, a: &Arr, source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, Arr> {
        if let Some(name) = a.as_str().strip_prefix(NAME_PREFIX)
            && target.len() == 1
            && closure_parts(&target[0]).is_some()
        {
            return map_name_operation(self.theory, name, source, target);
        }

        match a.as_str() {
            DEFER | RUN => OpenHypergraph::identity(map_objects(source)),
            COMPOSE => map_compose(source),
            TENSOR => map_tensor(source),
            LIFT => map_lift(source, target),
            _ => map_non_cmc_operation(a, source, target),
        }
    }

    fn map_arrow(&self, f: &OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: forget-closures is not a functor")
    }
}

/// Combine computed types with theory definitions to get [`AnnotatedTerm`]s
fn typed_definition(
    theory_id: &TheoryId,
    definition_name: &Operation,
    theory: &Theory,
    theory_definition_types: Option<&BTreeMap<Operation, Vec<Tree<(), Operation>>>>,
) -> Result<AnnotatedTerm, ForgetClosuresError> {
    let Theory::Theory { arrows, .. } = theory else {
        unreachable!("typed_definition only called on user theories");
    };
    let arrow = arrows
        .get(definition_name)
        .expect("definition should exist in current theory");
    let body = arrow
        .definition
        .clone()
        .ok_or_else(|| ForgetClosuresError::MissingDefinition {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;
    let mut body = body;
    body.quotient().ok();
    let labels = theory_definition_types
        .and_then(|types| types.get(definition_name))
        .cloned()
        .ok_or_else(|| ForgetClosuresError::MissingDefinitionTypes {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })?;
    body.with_nodes(|_| labels)
        .ok_or_else(|| ForgetClosuresError::NodeLabelCountMismatch {
            theory: theory_id.to_string(),
            definition: definition_name.to_string(),
        })
}

////////////////////////////////////////////////////////////////////////////////
/// Action of forget_closures on generating operations

// name.* operations map to the original operation, plus packers, with input wires 'bent around'
fn map_name_operation(
    theory: &Theory,
    name: &str,
    source: &[Obj],
    target: &[Obj],
) -> OpenHypergraph<Obj, Arr> {
    let [closure_type] = target else {
        panic!("name.* target should be a single closure-typed wire");
    };
    let (domain, codomain) =
        closure_parts(closure_type).expect("name.* target should be closure-typed");
    let operation: Operation = name
        .parse()
        .expect("stripped name.* operation should parse");
    let arrow = theory
        .get_arrow(&operation)
        .expect("name.* should refer to an arrow in the current theory");
    let operation_source = unpack_packed_object(domain, arrow.type_maps.0.targets.len());
    let operation_target = unpack_packed_object(codomain, arrow.type_maps.1.targets.len());
    let domain = map_objects(&operation_source);

    let cup = if source.is_empty() {
        cup(&domain)
    } else {
        let mapped_source = map_objects(source);
        assert_eq!(
            mapped_source, domain,
            "non-nullary name.* currently expects its source wires to match the closure domain"
        );
        duplicate_outputs(&mapped_source)
    };

    let id = OpenHypergraph::identity(domain.clone());
    let f = map_non_cmc_operation(&operation, &operation_source, &operation_target);
    cup.compose(&id.tensor(&f))
        .expect("name.* expansion should compose")
}

// Defines the action of forget_closures on non-CMC operations f:
// Φ ; f ; Φ⁻¹
fn map_non_cmc_operation(a: &Arr, source: &[Obj], target: &[Obj]) -> AnnotatedTerm {
    let pack = to_packer(source.to_vec());
    let operation = OpenHypergraph::singleton(
        a.clone(),
        forget_closures_in_objects(source),
        forget_closures_in_objects(target),
    );
    let unpack = to_unpacker(target.to_vec());

    pack.compose(&operation)
        .and_then(|packed| packed.compose(&unpack))
        .expect("regular operation adapters should compose")
}

fn map_compose(source: &[Obj]) -> OpenHypergraph<Obj, Arr> {
    let [lhs, rhs] = source else {
        panic!("compose should have two closure inputs");
    };
    let (a, b0) = closure_parts(lhs).expect("compose lhs should be closure-typed");
    let (b1, c) = closure_parts(rhs).expect("compose rhs should be closure-typed");

    let a = expand_object(a);
    let b0 = expand_object(b0);
    let b1 = expand_object(b1);
    let c = expand_object(c);
    assert_eq!(b0, b1, "compose intermediate object should agree");

    OpenHypergraph::identity(a)
        .tensor(&cap(&b0))
        .tensor(&OpenHypergraph::identity(c))
}

fn map_tensor(source: &[Obj]) -> OpenHypergraph<Obj, Arr> {
    let [lhs, rhs] = source else {
        panic!("tensor should have two closure inputs");
    };
    let (a0, b0) = closure_parts(lhs).expect("tensor lhs should be closure-typed");
    let (a1, b1) = closure_parts(rhs).expect("tensor rhs should be closure-typed");

    let a0 = expand_object(a0);
    let b0 = expand_object(b0);
    let a1 = expand_object(a1);
    let b1 = expand_object(b1);

    let mut result =
        OpenHypergraph::identity([a0.clone(), b0.clone(), a1.clone(), b1.clone()].concat());
    let a0_len = a0.len();
    let b0_len = b0.len();
    let a1_len = a1.len();
    let b1_len = b1.len();
    let order = (0..a0_len)
        .chain(a0_len + b0_len..a0_len + b0_len + a1_len)
        .chain(a0_len..a0_len + b0_len)
        .chain(a0_len + b0_len + a1_len..a0_len + b0_len + a1_len + b1_len)
        .map(NodeId)
        .collect();
    result.targets = order;
    result
}

fn map_lift(source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, Arr> {
    let [function_type] = source else {
        panic!("lift should have one function pointer input");
    };
    let [closure_type] = target else {
        panic!("lift should produce one closure output");
    };

    let (fn_domain, fn_codomain) = value_wrapped_function_parts(function_type)
        .expect("lift source should be value-wrapped function-typed");
    let (closure_domain, closure_codomain) =
        closure_parts(closure_type).expect("lift target should be closure-typed");

    assert_eq!(fn_domain, closure_domain, "lift domain should be preserved");
    assert_eq!(
        fn_codomain, closure_codomain,
        "lift codomain should be preserved"
    );

    let domain = expand_object(fn_domain);
    let function_pointer = vec![function_type.clone()];

    let prepare = cup(&domain).tensor(&OpenHypergraph::identity(function_pointer.clone()));
    let eval = map_non_cmc_operation(
        &op(EVAL),
        &[fn_domain.clone(), function_type.clone()],
        &[fn_codomain.clone()],
    );
    let finish = OpenHypergraph::identity(domain).tensor(&eval);

    prepare
        .compose(&finish)
        .expect("lift expansion should compose")
}

////////////////////////////////////////////////////////////////////////////////
/// Action of forget_closures on generating objects

fn expand_object(o: &Obj) -> Vec<Obj> {
    match o {
        Tree::Empty => vec![],
        Tree::Leaf(_, _) => vec![o.clone()],
        Tree::Node(op, _, children) if op.as_str() == UNIT_TYPE && children.is_empty() => vec![],
        Tree::Node(op, _, children) if op.as_str() == PRODUCT_TYPE => {
            children.iter().flat_map(expand_object).collect()
        }
        Tree::Node(op, _, children) if op.as_str() == CLOSURE_TYPE => {
            children.iter().flat_map(expand_object).collect()
        }
        _ => vec![o.clone()],
    }
}

fn map_objects(objects: &[Obj]) -> Vec<Obj> {
    objects.iter().flat_map(expand_object).collect()
}

fn forget_closures_in_object(object: &Obj) -> Vec<Obj> {
    match object {
        Tree::Node(op, _, children) if op.as_str() == CLOSURE_TYPE => children
            .iter()
            .flat_map(forget_closures_in_object)
            .collect(),
        _ => vec![object.clone()],
    }
}

fn forget_closures_in_objects(objects: &[Obj]) -> Vec<Obj> {
    objects.iter().flat_map(forget_closures_in_object).collect()
}

fn closure_parts(o: &Obj) -> Option<(&Obj, &Obj)> {
    parts(o, CLOSURE_TYPE)
}

fn function_parts(o: &Obj) -> Option<(&Obj, &Obj)> {
    parts(o, FUNCTION_TYPE)
}

fn value_wrapped_function_parts(o: &Obj) -> Option<(&Obj, &Obj)> {
    let inner = unwrap_value(o)?;
    function_parts(inner)
}

fn unwrap_value(o: &Obj) -> Option<&Obj> {
    let Tree::Node(op, _, children) = o else {
        return None;
    };
    if op.as_str() != VALUE_TYPE {
        return None;
    }
    let [inner] = children.as_slice() else {
        return None;
    };
    Some(inner)
}

fn parts<'a>(o: &'a Obj, op_name: &str) -> Option<(&'a Obj, &'a Obj)> {
    let Tree::Node(op, _, children) = o else {
        return None;
    };
    if op.as_str() != op_name {
        return None;
    }
    let [source, target] = children.as_slice() else {
        return None;
    };
    Some((source, target))
}

fn cup(object: &[Obj]) -> AnnotatedTerm {
    let mut result = OpenHypergraph::identity(object.to_vec());
    result.sources = vec![];
    result.targets = [result.targets.clone(), result.targets].concat();
    result
}

fn cap(object: &[Obj]) -> AnnotatedTerm {
    let mut result = OpenHypergraph::identity(object.to_vec());
    result.sources = [result.sources.clone(), result.sources].concat();
    result.targets = vec![];
    result
}

fn duplicate_outputs(object: &[Obj]) -> AnnotatedTerm {
    let mut result = OpenHypergraph::identity(object.to_vec());
    result.targets = [result.targets.clone(), result.targets].concat();
    result
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}

#[cfg(test)]
mod tests {
    use metacat::theory::{RawTheorySet, TheoryId, TheorySet};

    use super::*;

    fn object(name: &str) -> Obj {
        Tree::Node(op(name), 0, vec![])
    }

    fn product(left: Obj, right: Obj) -> Obj {
        Tree::Node(op("*"), 0, vec![left, right])
    }

    #[test]
    fn regular_operations_are_wrapped_in_packers_and_unpackers() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let d = object("D");
        let e = object("E");

        let mapped = map_non_cmc_operation(&op("f"), &[product(a, b), c], &[product(d, e)]);

        assert_eq!(
            mapped.hypergraph.edges,
            vec![op("*.intro"), op("f"), op("*.elim")]
        );
    }

    #[test]
    fn named_operation_uses_declared_arity_to_restore_product_arguments() {
        let raw = RawTheorySet::from_text(
            r#"
            (theory type nat {
              (arr * : 2 -> 1)
              (arr 1 : 0 -> 1)
              (arr => : 2 -> 1)
              (arr a : 0 -> 1)
              (arr b : 0 -> 1)
              (arr c : 0 -> 1)
              (arr d : 0 -> 1)
              (arr e : 0 -> 1)
            })

            (theory program type {
              (arr f : {({a b} *) c} -> {d e})
            })
            "#,
        )
        .expect("test theory should parse");
        let theories = TheorySet::from_raw(raw).expect("test theory should load");
        let theory = theories
            .theories
            .get(&TheoryId(op("program")))
            .expect("program theory should exist");

        let a = object("a");
        let b = object("b");
        let c = object("c");
        let d = object("d");
        let e = object("e");
        let closure = Tree::Node(
            op(CLOSURE_TYPE),
            0,
            vec![product(product(a, b), c), product(d, e)],
        );

        let mapped = map_name_operation(theory, "f", &[], &[closure]);

        assert_eq!(mapped.hypergraph.edges, vec![op("*.intro"), op("f")]);
        assert_eq!(mapped.sources.len(), 0);
        assert_eq!(mapped.targets.len(), 5);
    }

    #[test]
    fn lift_keeps_eval_at_arity_two() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let domain = product(a, b);
        let function = Tree::Node(
            op(VALUE_TYPE),
            0,
            vec![Tree::Node(
                op(FUNCTION_TYPE),
                0,
                vec![domain.clone(), c.clone()],
            )],
        );
        let closure = Tree::Node(op(CLOSURE_TYPE), 0, vec![domain.clone(), c]);

        let mapped = map_lift(&[function], &[closure]);
        let eval_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.as_str() == EVAL)
            .expect("lift expansion should contain eval");
        let eval = &mapped.hypergraph.adjacency[eval_index];

        assert_eq!(eval.sources.len(), 2);
        assert_eq!(eval.targets.len(), 1);
        assert_eq!(
            mapped.hypergraph.nodes[eval.sources[0].0], domain,
            "eval's first input should remain one packed object"
        );
    }
}
