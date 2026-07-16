//! Forget closure operations by wire bending.
use std::{collections::BTreeMap, fmt};

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
    check::{AnnotatedTerm, DefinitionTypes},
    nonstrict::{to_flatteners, to_packer, to_unpacker, unpack_packed_object},
    prefixes::{GENERATED_CONTEXT_PREFIX, NAME_PREFIX},
    report::TheoryTermMap,
    stdlib::constants::{
        COMPOSE, DEFER, EVAL, FN_HOM_TYPE, FN_REF_TYPE, LIFT, PRODUCT_ELIM, PRODUCT_INTRO,
        PRODUCT_TYPE, RUN, TENSOR, UNIT_ELIM, UNIT_INTRO, UNIT_TYPE, VALUE_TYPE,
    },
};

pub type Obj = Tree<(), Operation>;
pub type Arr = Operation;
pub type ClosureForgottenTerm = OpenHypergraph<Obj, ClosureForgotten<Arr>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClosureForgotten<A> {
    Operation(A),
    ClosureMarker,
}

impl<A: fmt::Display> fmt::Display for ClosureForgotten<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation(operation) => write!(f, "{operation}"),
            Self::ClosureMarker => write!(f, "!closure"),
        }
    }
}

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
) -> Result<TheoryTermMap<ClosureForgotten<Operation>>, ForgetClosuresError> {
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

impl Functor<Obj, Arr, Obj, ClosureForgotten<Arr>> for ForgetClosures<'_> {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        closure_forgotten_boundary(o).into_iter()
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> OpenHypergraph<Obj, ClosureForgotten<Arr>> {
        if let Some(name) = a.as_str().strip_prefix(NAME_PREFIX)
            && target.len() == 1
            && closure_parts(&target[0]).is_some()
        {
            return map_name_operation(self.theory, name, source, target);
        }

        if a.as_str().starts_with(GENERATED_CONTEXT_PREFIX) {
            return map_context_projection_operation(source, target);
        }

        match a.as_str() {
            PRODUCT_INTRO | PRODUCT_ELIM | UNIT_INTRO | UNIT_ELIM => {
                map_structural_operation(source, target)
            }
            DEFER | RUN => OpenHypergraph::identity(closure_forgotten_boundaries(source)),
            COMPOSE => map_compose(source),
            TENSOR => map_tensor(source),
            LIFT => map_lift(source, target),
            _ => map_non_cmc_operation(a, source, target),
        }
    }

    fn map_arrow(
        &self,
        f: &OpenHypergraph<Obj, Arr>,
    ) -> OpenHypergraph<Obj, ClosureForgotten<Arr>> {
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

fn map_structural_operation(source: &[Obj], target: &[Obj]) -> ClosureForgottenTerm {
    let source = closure_forgotten_boundaries(source);
    let target = closure_forgotten_boundaries(target);
    assert_eq!(
        source, target,
        "forgotten product/unit operation should have identical boundaries"
    );
    OpenHypergraph::identity(source)
}

fn map_context_projection_operation(source: &[Obj], target: &[Obj]) -> ClosureForgottenTerm {
    let mapped_source = closure_forgotten_boundaries(source);
    let mapped_target = closure_forgotten_boundaries(target);
    assert!(
        mapped_target.starts_with(&mapped_source),
        "context.closure.* should preserve the region inputs as its environment outputs"
    );

    let mut result: ClosureForgottenTerm = OpenHypergraph::identity(mapped_source.clone());
    let extra_targets = mapped_target[mapped_source.len()..]
        .iter()
        .map(|object| context_leaf_target(&mapped_source, &mut result, object))
        .collect::<Vec<_>>();
    result.targets.extend(extra_targets);
    result
}

fn context_leaf_target(
    mapped_source: &[Obj],
    result: &mut ClosureForgottenTerm,
    object: &Obj,
) -> NodeId {
    assert!(
        matches!(object, Tree::Leaf(_, _)),
        "context.closure.* extra outputs should only be context leaves for name.closure.*"
    );

    mapped_source
        .iter()
        .position(|source_object| source_object == object)
        .map(NodeId)
        .unwrap_or_else(|| result.new_node(object.clone()))
}

// name.* operations map to the original operation, plus packers, with input wires 'bent around'
fn map_name_operation(
    theory: &Theory,
    name: &str,
    source: &[Obj],
    target: &[Obj],
) -> ClosureForgottenTerm {
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
    let domain = closure_forgotten_boundaries(&operation_source);

    let cup = if source.is_empty() {
        cup(&domain)
    } else {
        let mapped_source = closure_forgotten_boundaries(source);
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
// adapt source ; f ; flatten target
fn map_non_cmc_operation(a: &Arr, source: &[Obj], target: &[Obj]) -> ClosureForgottenTerm {
    let (source, source_adapter) = closure_erased_source_adapter(source);
    let target = closure_erased_operation_objects(target);
    let operation = OpenHypergraph::singleton(
        ClosureForgotten::Operation(a.clone()),
        source,
        target.clone(),
    );
    let flatten = closure_forgotten_flatteners(&target);

    source_adapter
        .compose(&operation)
        .and_then(|adapted| adapted.compose(&flatten))
        .expect("regular operation adapters should compose")
}

fn map_compose(source: &[Obj]) -> ClosureForgottenTerm {
    let [lhs, rhs] = source else {
        panic!("compose should have two closure inputs");
    };
    let (a, b0) = closure_parts(lhs).expect("compose lhs should be closure-typed");
    let (b1, c) = closure_parts(rhs).expect("compose rhs should be closure-typed");

    let a = closure_forgotten_boundary(a);
    let b0 = closure_forgotten_boundary(b0);
    let b1 = closure_forgotten_boundary(b1);
    let c = closure_forgotten_boundary(c);
    assert_eq!(b0, b1, "compose intermediate object should agree");

    OpenHypergraph::identity(a)
        .tensor(&cap(&b0))
        .tensor(&OpenHypergraph::identity(c))
}

fn map_tensor(source: &[Obj]) -> ClosureForgottenTerm {
    let [lhs, rhs] = source else {
        panic!("tensor should have two closure inputs");
    };
    let (a0, b0) = closure_parts(lhs).expect("tensor lhs should be closure-typed");
    let (a1, b1) = closure_parts(rhs).expect("tensor rhs should be closure-typed");

    let a0 = closure_forgotten_boundary(a0);
    let b0 = closure_forgotten_boundary(b0);
    let a1 = closure_forgotten_boundary(a1);
    let b1 = closure_forgotten_boundary(b1);

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

fn map_lift(source: &[Obj], target: &[Obj]) -> ClosureForgottenTerm {
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

    let domain = closure_forgotten_boundary(fn_domain);
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

/// Interpret an object on the external boundary of the forget-closures functor.
fn closure_forgotten_boundary(o: &Obj) -> Vec<Obj> {
    match o {
        Tree::Empty => vec![],
        Tree::Leaf(_, _) => vec![o.clone()],
        Tree::Node(op, _, children) if op.as_str() == UNIT_TYPE && children.is_empty() => vec![],
        Tree::Node(op, _, children) if op.as_str() == PRODUCT_TYPE => children
            .iter()
            .flat_map(closure_forgotten_boundary)
            .collect(),
        Tree::Node(op, _, children) if op.as_str() == FN_HOM_TYPE => children
            .iter()
            .flat_map(closure_forgotten_boundary)
            .collect(),
        _ => vec![o.clone()],
    }
}

fn closure_forgotten_boundaries(objects: &[Obj]) -> Vec<Obj> {
    objects
        .iter()
        .flat_map(closure_forgotten_boundary)
        .collect()
}

/// Interpret objects at a non-CMC operation boundary, preserving product shape where needed so
/// product flatteners can adapt from/to the fully closure-forgotten boundary.
fn closure_erased_operation_objects(objects: &[Obj]) -> Vec<Obj> {
    objects
        .iter()
        .flat_map(closure_erased_operation_object)
        .collect()
}

fn closure_erased_operation_object(object: &Obj) -> Vec<Obj> {
    match object {
        Tree::Node(operation, _, children) if operation.as_str() == PRODUCT_TYPE => {
            let [left, right] = children.as_slice() else {
                panic!("product object should have exactly two children");
            };
            vec![Tree::Node(
                op(PRODUCT_TYPE),
                0,
                vec![
                    pack_closure_erased_operation_objects(left),
                    pack_closure_erased_operation_objects(right),
                ],
            )]
        }
        Tree::Node(op, _, children) if op.as_str() == FN_HOM_TYPE => children
            .iter()
            .flat_map(closure_erased_operation_object)
            .collect(),
        _ => vec![object.clone()],
    }
}

fn pack_closure_erased_operation_objects(object: &Obj) -> Obj {
    pack_objects(&closure_erased_operation_object(object))
}

fn pack_objects(objects: &[Obj]) -> Obj {
    match objects {
        [] => Tree::Node(op(UNIT_TYPE), 0, vec![]),
        [only] => only.clone(),
        [head, tail @ ..] => {
            Tree::Node(op(PRODUCT_TYPE), 0, vec![head.clone(), pack_objects(tail)])
        }
    }
}

#[derive(Clone, Copy)]
// Tracks whether source-side adapter construction is currently in a covariant
// or contravariant position. Crossing a closure domain flips polarity, so
// products there use flipped unpackers (`*.elim`) instead of normal packers
// (`*.intro`).
enum Polarity {
    Positive,
    Negative,
}

impl Polarity {
    fn flipped(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }
}

fn closure_erased_source_adapter(objects: &[Obj]) -> (Vec<Obj>, ClosureForgottenTerm) {
    objects
        .iter()
        .map(|object| source_adapter_object(object, Polarity::Positive))
        .fold(
            (Vec::new(), OpenHypergraph::empty()),
            |(mut objects, term), (next_objects, next_term)| {
                objects.extend(next_objects);
                (objects, term.tensor(&next_term))
            },
        )
}

fn source_adapter_object(object: &Obj, variance: Polarity) -> (Vec<Obj>, ClosureForgottenTerm) {
    match object {
        Tree::Node(operation, _, children) if operation.as_str() == PRODUCT_TYPE => {
            let [left, right] = children.as_slice() else {
                panic!("product object should have exactly two children");
            };

            let (left_object, left_adapter) = source_adapter_component(left, variance);
            let (right_object, right_adapter) = source_adapter_component(right, variance);
            let product = Tree::Node(
                op(PRODUCT_TYPE),
                0,
                vec![left_object.clone(), right_object.clone()],
            );
            let children = left_adapter.tensor(&right_adapter);
            let product_adapter = match variance {
                Polarity::Positive => OpenHypergraph::singleton(
                    ClosureForgotten::Operation(op(PRODUCT_INTRO)),
                    vec![left_object, right_object],
                    vec![product.clone()],
                ),
                Polarity::Negative => flip_boundaries(OpenHypergraph::singleton(
                    ClosureForgotten::Operation(op(PRODUCT_ELIM)),
                    vec![product.clone()],
                    vec![left_object, right_object],
                )),
            };

            (
                vec![product],
                children
                    .compose(&product_adapter)
                    .expect("product source adapter should compose"),
            )
        }
        Tree::Node(operation, _, children)
            if operation.as_str() == UNIT_TYPE && children.is_empty() =>
        {
            let unit = object.clone();
            let unit_adapter = match variance {
                Polarity::Positive => OpenHypergraph::singleton(
                    ClosureForgotten::Operation(op(UNIT_INTRO)),
                    vec![],
                    vec![unit.clone()],
                ),
                Polarity::Negative => flip_boundaries(OpenHypergraph::singleton(
                    ClosureForgotten::Operation(op(UNIT_ELIM)),
                    vec![unit.clone()],
                    vec![],
                )),
            };
            (vec![unit], unit_adapter)
        }
        Tree::Node(operation, _, children) if operation.as_str() == FN_HOM_TYPE => {
            let [source, target] = children.as_slice() else {
                panic!("closure object should have exactly two children");
            };
            let (mut source_objects, source_adapter) =
                source_adapter_object(source, variance.flipped());
            let (target_objects, target_adapter) = source_adapter_object(target, variance);
            let closure = object.clone();
            source_objects.extend(target_objects);
            let adapter = source_adapter
                .tensor(&target_adapter)
                .compose(&OpenHypergraph::singleton(
                    ClosureForgotten::ClosureMarker,
                    source_objects,
                    vec![closure.clone()],
                ))
                .expect("closure region adapter should compose");
            (vec![closure], adapter)
        }
        _ => (
            vec![object.clone()],
            OpenHypergraph::identity(vec![object.clone()]),
        ),
    }
}

fn source_adapter_component(object: &Obj, variance: Polarity) -> (Obj, ClosureForgottenTerm) {
    let (objects, adapter) = source_adapter_object(object, variance);
    let packed = pack_objects(&objects);
    let packer = match variance {
        Polarity::Positive => lift_operations(to_packer(objects)),
        Polarity::Negative => flip_boundaries(lift_operations(to_unpacker(objects))),
    };
    (
        packed,
        adapter
            .compose(&packer)
            .expect("packed source adapter component should compose"),
    )
}

fn flip_boundaries<O, A>(mut term: OpenHypergraph<O, A>) -> OpenHypergraph<O, A> {
    std::mem::swap(&mut term.sources, &mut term.targets);
    term
}

fn lift_operations(term: AnnotatedTerm) -> ClosureForgottenTerm {
    term.map_edges(ClosureForgotten::Operation)
}

fn closure_forgotten_flatteners(objects: &[Obj]) -> ClosureForgottenTerm {
    lift_operations(to_flatteners(objects))
}

fn closure_parts(o: &Obj) -> Option<(&Obj, &Obj)> {
    parts(o, FN_HOM_TYPE)
}

fn function_parts(o: &Obj) -> Option<(&Obj, &Obj)> {
    parts(o, FN_REF_TYPE)
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

fn cup(object: &[Obj]) -> ClosureForgottenTerm {
    let mut result = OpenHypergraph::identity(object.to_vec());
    result.sources = vec![];
    result.targets = [result.targets.clone(), result.targets].concat();
    result
}

fn cap(object: &[Obj]) -> ClosureForgottenTerm {
    let mut result = OpenHypergraph::identity(object.to_vec());
    result.sources = [result.sources.clone(), result.sources].concat();
    result.targets = vec![];
    result
}

fn duplicate_outputs(object: &[Obj]) -> ClosureForgottenTerm {
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

    fn source_types<A>(term: &OpenHypergraph<Obj, A>) -> Vec<Obj> {
        term.sources
            .iter()
            .map(|node| term.hypergraph.nodes[node.0].clone())
            .collect()
    }

    fn target_types<A>(term: &OpenHypergraph<Obj, A>) -> Vec<Obj> {
        term.targets
            .iter()
            .map(|node| term.hypergraph.nodes[node.0].clone())
            .collect()
    }

    fn assert_closure_forgotten_boundaries<A>(
        term: &OpenHypergraph<Obj, A>,
        source: &[Obj],
        target: &[Obj],
    ) {
        assert_eq!(source_types(term), closure_forgotten_boundaries(source));
        assert_eq!(target_types(term), closure_forgotten_boundaries(target));
    }

    fn region_op(name: &str) -> ClosureForgotten<Operation> {
        ClosureForgotten::Operation(op(name))
    }

    #[test]
    fn structural_operations_with_closures_map_to_identities() {
        let a = object("A");
        let b = object("B");
        let x = object("X");
        let closure = Tree::Node(op(FN_HOM_TYPE), 0, vec![a.clone(), b.clone()]);
        let packed = product(closure.clone(), x.clone());

        let intro =
            map_structural_operation(&[closure.clone(), x.clone()], std::slice::from_ref(&packed));
        let elim = map_structural_operation(&[packed], &[closure, x.clone()]);

        for mapped in [intro, elim] {
            assert!(mapped.hypergraph.edges.is_empty());
            assert_eq!(source_types(&mapped), vec![a.clone(), b.clone(), x.clone()]);
            assert_eq!(target_types(&mapped), vec![a.clone(), b.clone(), x.clone()]);
        }
    }

    #[test]
    fn regular_operations_are_wrapped_in_flatteners() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let d = object("D");
        let e = object("E");

        let mapped = map_non_cmc_operation(&op("f"), &[product(a, b), c], &[product(d, e)]);

        assert_eq!(
            mapped.hypergraph.edges,
            vec![region_op("*.intro"), region_op("f"), region_op("*.elim")]
        );
    }

    #[test]
    fn regular_operation_adapter_flattens_boundary_without_repacking_operation_arity() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let d = object("D");
        let e = object("E");
        let f = object("F");
        let g = object("G");
        let h = object("H");
        let i = object("I");
        let source = vec![
            product(product(a.clone(), b.clone()), c.clone()),
            product(d.clone(), e.clone()),
        ];
        let target = vec![product(f.clone(), product(g.clone(), h.clone())), i.clone()];

        let mapped = map_non_cmc_operation(&op("f"), &source, &target);
        let operation_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.to_string() == "f")
            .expect("adapter should contain the original operation");
        let operation = &mapped.hypergraph.adjacency[operation_index];

        assert_eq!(
            source_types(&mapped),
            vec![a.clone(), b.clone(), c.clone(), d.clone(), e.clone()]
        );
        assert_eq!(
            target_types(&mapped),
            vec![f.clone(), g.clone(), h.clone(), i.clone()]
        );
        assert_eq!(
            operation
                .sources
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            source,
            "f should still see its declared top-level source objects"
        );
        assert_eq!(
            operation
                .targets
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            target,
            "f should still produce its declared top-level target objects"
        );
    }

    #[test]
    fn regular_operation_adapter_forgets_closures_nested_under_products() {
        let ix = object("Ix");
        let f32 = object("F32");
        let arg = object("Arg");
        let out = object("Out");
        let closure = Tree::Node(op(FN_HOM_TYPE), 0, vec![ix.clone(), f32.clone()]);
        let source = vec![product(closure, arg.clone())];

        let mapped = map_non_cmc_operation(&op("f"), &source, std::slice::from_ref(&out));
        let operation_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.to_string() == "f")
            .expect("adapter should contain the original operation");
        let operation = &mapped.hypergraph.adjacency[operation_index];

        assert_eq!(source_types(&mapped), vec![ix.clone(), f32.clone(), arg]);
        assert_eq!(target_types(&mapped), vec![out.clone()]);
        assert_eq!(
            operation
                .sources
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![product(
                Tree::Node(op(FN_HOM_TYPE), 0, vec![ix, f32]),
                object("Arg")
            )],
            "f should see a bracketed closure object nested inside product shape"
        );
        assert_eq!(
            operation
                .targets
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![out]
        );
    }

    #[test]
    fn source_closure_domains_use_flipped_unpackers() {
        let a = object("A");
        let b = object("B");
        let c = object("C");
        let out = object("Out");
        let domain = product(a.clone(), b.clone());
        let closure = Tree::Node(op(FN_HOM_TYPE), 0, vec![domain.clone(), c.clone()]);

        let mut mapped = map_non_cmc_operation(
            &op("reduce-like"),
            std::slice::from_ref(&closure),
            std::slice::from_ref(&out),
        );
        mapped
            .quotient()
            .expect("test term should quotient after adapter composition");

        let operation_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.to_string() == "reduce-like")
            .expect("adapter should contain the original operation");
        let operation = &mapped.hypergraph.adjacency[operation_index];
        let elim_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.to_string() == "*.elim")
            .expect("closure domain product should be unpacked");
        let elim = &mapped.hypergraph.adjacency[elim_index];
        let closure_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| matches!(operation, ClosureForgotten::ClosureMarker))
            .expect("closure source should be bracketed");
        let closure_edge = &mapped.hypergraph.adjacency[closure_index];

        assert_eq!(source_types(&mapped), vec![a.clone(), b.clone(), c.clone()]);
        assert_eq!(target_types(&mapped), vec![out]);
        assert_eq!(
            operation
                .sources
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![closure.clone()],
            "the operation should see the bracketed closure input"
        );
        assert_eq!(
            closure_edge
                .sources
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![domain.clone(), c],
            "the bracket should receive the erased closure domain and codomain"
        );
        assert_eq!(
            elim.sources,
            vec![closure_edge.sources[0]],
            "*.elim source should be the bracket-side closure-domain product"
        );
        assert_eq!(
            elim.targets
                .iter()
                .map(|node| mapped.hypergraph.nodes[node.0].clone())
                .collect::<Vec<_>>(),
            vec![a, b],
            "*.elim targets should be the new subdiagram source boundary wires"
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
            op(FN_HOM_TYPE),
            0,
            vec![product(product(a, b), c), product(d, e)],
        );

        let mapped = map_name_operation(theory, "f", &[], std::slice::from_ref(&closure));

        assert_eq!(
            mapped.hypergraph.edges,
            vec![region_op("*.intro"), region_op("f")]
        );
        assert_eq!(mapped.sources.len(), 0);
        assert_eq!(
            target_types(&mapped),
            closure_forgotten_boundaries(std::slice::from_ref(&closure))
        );
    }

    #[test]
    fn mapped_generators_use_closure_forgotten_boundaries() {
        let a0 = object("A0");
        let a1 = object("A1");
        let b0 = object("B0");
        let b1 = object("B1");
        let c = object("C");
        let d = object("D");
        let closure0 = Tree::Node(op(FN_HOM_TYPE), 0, vec![a0.clone(), b0.clone()]);
        let closure1 = Tree::Node(op(FN_HOM_TYPE), 0, vec![b0.clone(), c.clone()]);
        let closure_2 = Tree::Node(op(FN_HOM_TYPE), 0, vec![a1.clone(), b1.clone()]);

        assert_closure_forgotten_boundaries(
            &map_non_cmc_operation(
                &op("f"),
                &[product(closure0.clone(), d.clone())],
                &[c.clone()],
            ),
            &[product(closure0.clone(), d.clone())],
            &[c.clone()],
        );

        let context = Tree::Leaf(0, ());
        assert_closure_forgotten_boundaries(
            &map_context_projection_operation(
                &[context.clone(), closure0.clone()],
                &[context.clone(), closure0.clone(), context.clone()],
            ),
            &[context.clone(), closure0.clone()],
            &[context, closure0.clone(), Tree::Leaf(0, ())],
        );

        assert_closure_forgotten_boundaries(
            &map_compose(&[closure0.clone(), closure1.clone()]),
            &[closure0.clone(), closure1.clone()],
            &[Tree::Node(op(FN_HOM_TYPE), 0, vec![a0.clone(), c.clone()])],
        );

        assert_closure_forgotten_boundaries(
            &map_tensor(&[closure0.clone(), closure_2.clone()]),
            &[closure0.clone(), closure_2],
            &[Tree::Node(
                op(FN_HOM_TYPE),
                0,
                vec![product(a0.clone(), a1), product(b0, b1)],
            )],
        );

        let function = Tree::Node(
            op(VALUE_TYPE),
            0,
            vec![Tree::Node(
                op(FN_REF_TYPE),
                0,
                vec![product(closure1.clone(), d.clone()), c.clone()],
            )],
        );
        let lifted = Tree::Node(op(FN_HOM_TYPE), 0, vec![product(closure1, d), c]);

        assert_closure_forgotten_boundaries(
            &map_lift(&[function.clone()], std::slice::from_ref(&lifted)),
            &[function],
            &[lifted],
        );
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
                op(FN_REF_TYPE),
                0,
                vec![domain.clone(), c.clone()],
            )],
        );
        let closure = Tree::Node(op(FN_HOM_TYPE), 0, vec![domain.clone(), c]);

        let mapped = map_lift(&[function], &[closure]);
        let eval_index = mapped
            .hypergraph
            .edges
            .iter()
            .position(|operation| operation.to_string() == EVAL)
            .expect("lift expansion should contain eval");
        let eval = &mapped.hypergraph.adjacency[eval_index];

        assert_eq!(eval.sources.len(), 2);
        assert_eq!(eval.targets.len(), 1);
        assert_eq!(
            mapped.hypergraph.nodes[eval.sources[0].0], domain,
            "eval's first input should remain one packed object"
        );
    }

    #[test]
    fn lift_domain_with_nested_closure_composes_against_eval_adapter_boundary() {
        let ix = object("Ix");
        let f32 = object("F32");
        let arg = object("Arg");
        let out = object("Out");
        let closure = Tree::Node(op(FN_HOM_TYPE), 0, vec![ix.clone(), f32.clone()]);
        let domain = product(product(closure.clone(), closure), arg);
        let function = Tree::Node(
            op(VALUE_TYPE),
            0,
            vec![Tree::Node(
                op(FN_REF_TYPE),
                0,
                vec![domain.clone(), out.clone()],
            )],
        );
        let lifted = Tree::Node(op(FN_HOM_TYPE), 0, vec![domain, out]);

        let mapped = map_lift(&[function], &[lifted]);

        assert_eq!(
            target_types(&mapped),
            vec![
                ix.clone(),
                f32.clone(),
                ix,
                f32,
                object("Arg"),
                object("Out")
            ]
        );
    }
}
