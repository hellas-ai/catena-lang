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

#[derive(Clone)]
struct ForgetClosures;

impl Functor<Obj, Arr, Obj, Arr> for ForgetClosures {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        expand_object(o).into_iter()
    }

    fn map_operation(&self, a: &Arr, source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, Arr> {
        if let Some(name) = a.as_str().strip_prefix(NAME_PREFIX)
            && target.len() == 1
            && closure_parts(&target[0]).is_some()
        {
            return map_name_operation(name, source, target);
        }

        match a.as_str() {
            DEFER | RUN => OpenHypergraph::identity(map_objects(source)),
            COMPOSE => map_compose(source),
            TENSOR => map_tensor(source),
            LIFT => map_lift(source, target),
            _ => OpenHypergraph::singleton(a.clone(), map_objects(source), map_objects(target)),
        }
    }

    fn map_arrow(&self, f: &OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: forget-closures is not a functor")
    }
}

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
        for (definition_name, arrow) in arrows {
            let Some(_) = &arrow.definition else {
                continue;
            };

            let typed =
                typed_definition(theory_id, definition_name, theory, theory_definition_types)?;
            transformed.insert(definition_name.clone(), ForgetClosures.map_arrow(&typed));
        }

        if !transformed.is_empty() {
            output.insert(theory_id.clone(), transformed);
        }
    }

    Ok(output)
}

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

fn map_name_operation(name: &str, source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, Arr> {
    let [closure_type] = target else {
        panic!("name.* target should be a single closure-typed wire");
    };
    let (domain, codomain) =
        closure_parts(closure_type).expect("name.* target should be closure-typed");
    let domain = expand_object(domain);
    let codomain = expand_object(codomain);

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
    let f = OpenHypergraph::singleton(
        name.parse()
            .expect("stripped name.* operation should parse"),
        domain,
        codomain,
    );
    cup.compose(&id.tensor(&f))
        .expect("name.* expansion should compose")
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
    let codomain = expand_object(fn_codomain);
    let function_pointer = vec![function_type.clone()];

    let prepare = cup(&domain).tensor(&OpenHypergraph::identity(function_pointer.clone()));
    let eval = OpenHypergraph::singleton(
        op(EVAL),
        [domain.clone(), function_pointer].concat(),
        codomain,
    );
    let finish = OpenHypergraph::identity(domain).tensor(&eval);

    prepare
        .compose(&finish)
        .expect("lift expansion should compose")
}

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
