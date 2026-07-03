use hexpr::{Hexpr, Operation};
use metacat::{
    theory::{RawTheorySet, Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::{NodeId, OpenHypergraph};

use crate::{
    check::{AnnotatedTerm, DefinitionTypes, check},
    closure::{
        body::closure_body,
        convert::{ConvertError, convert},
        extract::extract_region,
        region::{ClosureRegionError, Obj, closure_region},
        theory::convert_theory,
    },
    elaborate::elaborate,
    prefixes::GENERATED_COPY_PREFIX,
    stdlib::{
        self,
        constants::{FN_HOM_TYPE, PRODUCT_TYPE, UNIT_TYPE},
    },
};

#[test]
fn identity_wire_has_no_closure_work_to_do() {
    let definition = annotated_program_definition(
        r#"
        (def program id : [a] -> [a] = [x])
        "#,
        "id",
    );

    let closure_wires = closure_wires(&definition);
    let regions =
        closure_region(&definition, &closure_wires).expect("region discovery should succeed");
    let converted =
        convert(&op("id"), &definition, &closure_wires).expect("conversion should succeed");

    // No closure regions appear, none converted
    assert_eq!(regions.len(), 0);
    assert_eq!(converted.closures.len(), 0);

    // input definition is (approx.) the same
    assert_same_definition_interface(&converted.definition, &definition);
}

#[test]
fn convert_rejects_explicit_non_closure_wire() {
    let definition = annotated_program_definition(
        r#"
        (def program id : [a] -> [a] = [x])
        "#,
        "id",
    );
    let non_closure_wire = definition.sources[0];

    let error = convert(&op("id"), &definition, &[non_closure_wire])
        .expect_err("non-closure wire should be rejected");

    assert!(matches!(
        error,
        ConvertError::Region(ClosureRegionError::NotClosureTyped { wire })
            if wire == non_closure_wire.0
    ));
}

#[test]
fn deferred_bool_id_closure_converts_through_each_stage() {
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
    let original_target = definition.targets[0];

    // Find the (single) closure region at the output node
    let [region] = closure_region(&definition, &[original_target])
        .expect("region discovery should succeed")
        .try_into()
        .expect("expected one closure region");
    assert_eq!(region.edges.len(), 4);

    // Extract the region into its own AnnotatedTerm, check it has expected sources (deferred data)
    // and targets (the closure type)
    let extracted = extract_region(&definition, &region).expect("region extraction should succeed");
    assert_eq!(extracted.hypergraph.edges.len(), 4);
    assert_eq!(
        interface_types(&extracted, &extracted.sources),
        interface_types(&definition, &region.leaf_inputs)
    );
    assert_eq!(
        interface_types(&extracted, &extracted.targets),
        vec![region.closure_type.clone()]
    );

    // Transform the closure into an arrow of type (X ● A -> B)
    // where here: X = (val bool), A = 1, B = val bool.
    let body = closure_body(&extracted).expect("closure body construction should succeed");
    assert_eq!(
        body.hypergraph.edges.len(),
        extracted.hypergraph.edges.len() + 3
    );
    assert_eq!(
        interface_types(&body, &body.sources),
        vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![])]
    );
    assert_eq!(
        interface_types(&body, &body.targets),
        vec![obj("val", vec![obj("bool", vec![])])]
    );

    // Run the 'convert' method
    let definition_name = op("run-bool-id");
    let converted = convert(&definition_name, &definition, &[original_target])
        .expect("conversion should succeed");

    // There should be one converted closure
    assert_eq!(converted.closures.len(), 1);

    // Check the closure type is (A => B)
    let converted_closure = &converted.closures[0];
    assert_eq!(converted_closure.node, original_target);

    // Check name is expected (closure.original-def.node_id)
    assert_eq!(
        converted_closure.name(&definition_name),
        op(&format!("closure.run-bool-id.{}", original_target.0))
    );

    // Check equivalent to body
    assert_eq!(
        converted_closure.term.hypergraph.edges.len(),
        body.hypergraph.edges.len()
    );

    // Check X = val(bool), A = 1, B = val(bool)
    assert_eq!(
        converted_closure.type_info.environment,
        obj("val", vec![obj("bool", vec![])])
    );
    assert_eq!(converted_closure.type_info.domain, obj("1", vec![]));
    assert_eq!(
        converted_closure.type_info.codomain,
        obj("val", vec![obj("bool", vec![])])
    );

    // Check converted definition has type
    //         X ● (X * A -> B)
    // val(bool) ● (val(bool) * 1 -> val(bool))
    assert_converted_definition(
        &converted.definition,
        5,
        vec![obj("val", vec![obj("bool", vec![])])],
        vec![
            obj("val", vec![obj("bool", vec![])]),
            function_pointer_type(
                vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![])],
                vec![obj("val", vec![obj("bool", vec![])])],
            ),
        ],
    );
    assert!(
        converted
            .definition
            .hypergraph
            .edges
            .iter()
            .any(|operation| operation.as_str()
                == format!(
                    "{GENERATED_COPY_PREFIX}closure.run-bool-id.{}.0",
                    original_target.0
                )),
        "converted definition should split the captured environment before naming the closure"
    );

    // Verify that original definition uses the *name* of the closure conversion
    assert!(
        converted
            .definition
            .hypergraph
            .edges
            .iter()
            .any(|operation| operation.as_str()
                == format!("name.closure.run-bool-id.{}", original_target.0))
    );
}

#[test]
fn closure_body_unpacker_reproduces_product_typed_environment_wires() {
    let width = obj("width", vec![]);
    let buffer = obj("buffer", vec![]);
    let row = obj("row", vec![]);
    let col = obj("col", vec![]);
    let scale = obj("scale", vec![]);
    let env_product = binary_product(width.clone(), buffer.clone());
    let idx = binary_product(row.clone(), col.clone());
    let captured = vec![env_product.clone(), idx.clone(), scale.clone()];
    let domain = obj("argument", vec![]);
    let codomain = obj("output", vec![]);
    let closure = obj(FN_HOM_TYPE, vec![domain.clone(), codomain.clone()]);

    let mut extracted: AnnotatedTerm = OpenHypergraph::empty();
    extracted.sources = captured
        .iter()
        .map(|object| extracted.new_node(object.clone()))
        .collect();
    extracted.targets = vec![extracted.new_node(closure)];

    let body = closure_body(&extracted).expect("closure body construction should succeed");
    let body_sources = interface_types(&body, &body.sources);
    let expected_environment = right_associated_product(&captured);

    assert_eq!(
        body_sources,
        vec![expected_environment, domain],
        "closure body should receive one packed environment plus the argument"
    );

    let product_elims = body
        .hypergraph
        .edges
        .iter()
        .filter(|operation| operation.as_str() == "*.elim")
        .count();
    assert_eq!(
        product_elims,
        captured.len() - 1,
        "environment unpacker should split only top-level captured wires"
    );

    let unpacked_environment_targets = body
        .hypergraph
        .adjacency
        .iter()
        .zip(&body.hypergraph.edges)
        .filter(|(_, operation)| operation.as_str() == "*.elim")
        .flat_map(|(edge, _)| edge.targets.iter())
        .filter_map(|node| {
            let ty = &body.hypergraph.nodes[node.0];
            captured.contains(ty).then_some(ty.clone())
        })
        .collect::<Vec<_>>();
    assert!(
        captured
            .iter()
            .all(|ty| unpacked_environment_targets.contains(ty)),
        "unpacker should reproduce product-typed captured wires themselves"
    );
}

#[test]
fn converted_closure_name_keeps_free_variable_input() {
    // Build the smallest closure region that depends on a free variable.
    //
    // The hand-built `name.manual-ix-n-to-u64` edge stands for a named closure
    // family indexed by `n`. Supplying the free length parameter `n` produces a
    // concrete closure of type `ix n => u64`.
    //
    // Types used in the hand-built graph:
    //
    //   n            type-level length parameter
    //   val(ix n)    runtime index into a collection of length n
    //   val(u64)     runtime u64 output
    //   ix n => u64  closure returned by name.manual-ix-n-to-u64
    let length_parameter_n = Tree::Leaf(0, ());
    let index_value_at_n = obj("val", vec![obj("ix", vec![length_parameter_n.clone()])]);
    let u64_value = obj("val", vec![obj("u64", vec![])]);
    let producer_closure_type = obj(
        FN_HOM_TYPE,
        vec![index_value_at_n.clone(), u64_value.clone()],
    );

    // Wires and edges:
    //
    //   n -- name.manual-ix-n-to-u64 --> (ix n => u64)
    //
    // `free_n` indexes which closure name should be produced. Therefore the
    // replacement `name.closure.reduce-n.*` must still receive `free_n`.
    let mut definition = AnnotatedTerm::empty();
    let free_n = definition.new_node(length_parameter_n);
    let closure = definition.new_node(producer_closure_type);

    definition.new_edge(op("name.manual-ix-n-to-u64"), (vec![free_n], vec![closure]));
    definition.sources = vec![free_n];
    definition.targets = vec![closure];

    let constructed = crate::hexpr::term_to_hexpr(&definition);
    let expected_construction: Hexpr =
        "([w0 . ] ([ . w0] name.manual-ix-n-to-u64 [w1 . ]) [ . w1])"
            .parse()
            .expect("expected constructed definition Hexpr should parse");
    assert_eq!(
        constructed, expected_construction,
        "test setup should construct w0=n |- name.manual-ix-n-to-u64"
    );

    // Closure conversion replaces the closure-producing region with an explicit
    // environment and a generated closure name. Since the manual closure name
    // depended on `n`, the generated `name.closure.reduce-n.*` must preserve the
    // same dependency instead of becoming nullary.
    let converted =
        convert(&op("reduce-n"), &definition, &[closure]).expect("conversion should succeed");
    let generated_closure = converted
        .closures
        .first()
        .expect("conversion should generate a closure body");
    assert_eq!(
        interface_types(&generated_closure.term, &generated_closure.term.sources),
        vec![
            Tree::Leaf(0, ()),
            obj("val", vec![obj("ix", vec![Tree::Leaf(0, ())])])
        ],
        "generated closure body should still depend on free variable n"
    );

    let mut converted_definition = converted.definition.clone();
    converted_definition
        .quotient()
        .expect("quotient should succeed");

    let converted_hexpr = crate::hexpr::term_to_hexpr(&converted_definition);
    let expected_converted: Hexpr = format!(
        "([w0 . ] ([ . w0] {GENERATED_COPY_PREFIX}closure.reduce-n.1.0 [w1 w2 . ]) \
         ([ . w2] name.closure.reduce-n.1 [w3 . ]) [ . w1 w3])"
    )
    .as_str()
    .parse()
    .expect("expected converted definition Hexpr should parse");
    assert_eq!(
        converted_hexpr, expected_converted,
        "closure conversion should split n, keep one copy as the environment, \
         and pass the other copy to the generated closure name"
    );

    let name_edge = converted_definition
        .hypergraph
        .edges
        .iter()
        .zip(&converted_definition.hypergraph.adjacency)
        .find(|(operation, _)| operation.as_str().starts_with("name.closure.reduce-n."))
        .map(|(_, edge)| edge)
        .expect("converted definition should generate a closure name edge");

    // `name.closure.reduce-n.*` should no longer be nullary: it receives the
    // copied `n` input produced by the split above, and returns only the
    // function pointer. The environment remains the other split output.
    assert_eq!(
        name_edge
            .sources
            .iter()
            .map(|node| converted_definition.hypergraph.nodes[node.0].clone())
            .collect::<Vec<_>>(),
        vec![Tree::Leaf(0, ())],
        "generated closure name should consume the free variable n"
    );
    assert_eq!(
        name_edge.targets,
        vec![converted_definition.targets[1]],
        "generated closure name should only produce the function pointer"
    );
}

#[test]
fn theory_conversion_converts_if_closure_arguments() {
    let (theory_set, definition_types) = theories_with(
        r#"
        (def program f32.id : (f32 val) -> (f32 val) = [x])
        (def program if-id-neg : {(bool val) (f32 val)} -> (f32 val) = ([b x.]
          {(name.f32.id lift) (name.f32.neg lift) [.b] [.x]} bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));

    let converted =
        convert_theory(&theory_set, &definition_types, &program).expect("theory should convert");
    let mut converted_set = theory_set.clone();
    converted_set
        .theories
        .insert(program.clone(), converted.clone());
    let converted_definition_types =
        check(&converted_set).expect("converted theory should still typecheck");
    let forgotten = crate::pass::forget_closures::run(&converted_set, &converted_definition_types)
        .expect("forget-closures should erase converted closure structure");
    let forgotten_if_id_neg = forgotten
        .get(&program)
        .and_then(|definitions| definitions.get(&op("if-id-neg")))
        .expect("forgotten closures should include converted if-id-neg");
    assert!(
        forgotten_if_id_neg
            .hypergraph
            .edges
            .iter()
            .all(|operation| !operation.as_str().starts_with(GENERATED_COPY_PREFIX)),
        "copy.closure.* should be erased before codegen"
    );

    let Theory::Theory { arrows, .. } = converted else {
        panic!("program should be a theory");
    };

    let if_id_neg = arrows
        .get(&op("if-id-neg"))
        .expect("converted original definition should exist");
    let converted_body = if_id_neg
        .definition
        .as_ref()
        .expect("converted original definition should have a body");
    assert!(
        converted_body
            .hypergraph
            .edges
            .iter()
            .any(|operation| operation.as_str() == "bool.ifc")
    );
    assert!(
        converted_body
            .hypergraph
            .edges
            .iter()
            .any(|operation| operation.as_str().starts_with("name.closure.if-id-neg."))
    );

    let closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("closure.if-id-neg."))
        .collect::<Vec<_>>();
    let name_closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("name.closure.if-id-neg."))
        .collect::<Vec<_>>();
    assert_eq!(closure_names.len(), 2);
    assert_eq!(name_closure_names.len(), 2);
}

#[test]
fn theory_conversion_converts_if_id_neg_example_end_to_end() {
    let (theory_set, definition_types) = theories_with(
        r#"
        (def program f32.id : (f32 val) -> (f32 val) = [x]) # specialised f32.id
        (def program if-id-neg : {(bool val) (f32 val)} -> (f32 val) = ([b x.]
          {(name.f32.id lift) (name.f32.neg lift) [.b] [.x]} bool.if
        ))
        "#,
    );
    let program = TheoryId(op("program"));

    let converted =
        convert_theory(&theory_set, &definition_types, &program).expect("theory should convert");
    let Theory::Theory { arrows, .. } = converted else {
        panic!("program should be a theory");
    };

    let if_id_neg = arrows
        .get(&op("if-id-neg"))
        .expect("converted original definition should exist");
    let if_id_neg_body = if_id_neg
        .definition
        .as_ref()
        .expect("converted original definition should have a body");
    assert_eq!(if_id_neg.type_maps.0.targets.len(), 2);
    assert_eq!(if_id_neg.type_maps.1.targets.len(), 1);
    assert_operation_count(if_id_neg_body, "bool.ifc", 1);
    assert_operation_count(if_id_neg_body, "bool.if", 0);

    let closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("closure.if-id-neg."))
        .cloned()
        .collect::<Vec<_>>();
    let name_closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("name.closure.if-id-neg."))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(closure_names.len(), 2);
    assert_eq!(name_closure_names.len(), 2);

    for closure_name in &closure_names {
        let closure = arrows
            .get(closure_name)
            .expect("generated closure arrow should exist");
        assert!(
            closure.definition.is_some(),
            "generated closure arrow {closure_name} should have a definition"
        );
        assert!(
            closure.raw.definition.is_some(),
            "generated closure arrow {closure_name} should have a raw definition"
        );
        assert_eq!(closure.type_maps.0.targets.len(), 2);
        assert_eq!(closure.type_maps.1.targets.len(), 1);
    }

    for name_closure_name in &name_closure_names {
        let name_closure = arrows
            .get(name_closure_name)
            .expect("generated name arrow should exist");
        assert!(
            name_closure.definition.is_none(),
            "generated name arrow {name_closure_name} should be a declaration"
        );
        assert!(name_closure.raw.definition.is_none());
        assert_eq!(name_closure.type_maps.0.targets.len(), 0);
        assert_eq!(name_closure.type_maps.1.targets.len(), 1);
    }

    for name_closure_name in &name_closure_names {
        assert!(
            if_id_neg_body
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation == name_closure_name),
            "converted if-id-neg should refer to {name_closure_name}"
        );
    }
}

#[test]
fn convert_parallel_regions_with_crossed_node_and_edge_order() {
    let bool_value = obj("val", vec![obj("bool", vec![])]);
    let unit = obj("1", vec![]);
    let unit_to_bool = obj("=>", vec![unit, bool_value.clone()]);

    // Metacat-ish shape:
    //
    //   {({x defer} (name.test.id) compose) (y defer)} test.use-two
    //
    // Both inputs to `test.use-two` are closure regions. The graph is built by
    // hand so the first region has higher node ids but lower edge ids than the
    // second region; rewriting in node-id order deletes/relabels edges before
    // the later region's recorded edge ids are consumed. The stale id used to
    // show up as `Rewrite(RegionEdgeOutOfBounds { edge: 4 })`.
    let mut definition = AnnotatedTerm::empty();
    let second_region_input = definition.new_node(bool_value.clone());
    let second_region_closure = definition.new_node(unit_to_bool.clone());
    let output = definition.new_node(bool_value.clone());
    let first_region_input = definition.new_node(bool_value);
    let first_region_deferred = definition.new_node(unit_to_bool.clone());
    let first_region_named = definition.new_node(unit_to_bool.clone());
    let first_region_closure = definition.new_node(unit_to_bool);

    definition.new_edge(
        op("defer"),
        (vec![first_region_input], vec![first_region_deferred]),
    );
    definition.new_edge(op("name.test.id"), (vec![], vec![first_region_named]));
    definition.new_edge(
        op("compose"),
        (
            vec![first_region_deferred, first_region_named],
            vec![first_region_closure],
        ),
    );
    definition.new_edge(
        op("test.use-two"),
        (
            vec![first_region_closure, second_region_closure],
            vec![output],
        ),
    );
    definition.new_edge(
        op("defer"),
        (vec![second_region_input], vec![second_region_closure]),
    );
    definition.sources = vec![second_region_input, first_region_input];
    definition.targets = vec![output];

    convert(
        &op("parallel-closures"),
        &definition,
        &[first_region_closure, second_region_closure],
    )
    .expect("parallel closure conversion should succeed");
}

#[test]
fn theory_conversion_converts_reduce_closure_arguments() {
    let (theory_set, definition_types) = theories_with(
        r#"
        (def program u64.one-at :
          ([n.] ([.n] ix val))
          ->
          ([n.] (u64 val))
        = ([i.] u64.one))

        (def program reduce-ones :
          []
          ->
          (u64 val)
        = (
          {
            const.u64.0x0000000000000000
            (name.u64.add lift)
            ((u64.zero :.param) name.u64.one-at lift)
            u64.zero
          }
          reduce
        ))
        "#,
    );
    let program = TheoryId(op("program"));

    let converted =
        convert_theory(&theory_set, &definition_types, &program).expect("theory should convert");

    let Theory::Theory { arrows, .. } = converted else {
        panic!("program should be a theory");
    };

    let reduce_ones = arrows
        .get(&op("reduce-ones"))
        .expect("converted original definition should exist");
    let reduce_ones_body = reduce_ones
        .definition
        .as_ref()
        .expect("converted original definition should have a body");

    assert_operation_count(reduce_ones_body, "reducec", 1);
    assert_operation_count(reduce_ones_body, "reduce", 0);

    let closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("closure.reduce-ones."))
        .cloned()
        .collect::<Vec<_>>();
    let name_closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("name.closure.reduce-ones."))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(closure_names.len(), 2);
    assert_eq!(name_closure_names.len(), 2);

    for name_closure_name in &name_closure_names {
        assert!(
            reduce_ones_body
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation == name_closure_name),
            "converted reduce-ones should refer to {name_closure_name}"
        );
    }
}

#[test]
fn theory_conversion_declares_context_dependent_copy_arrows() {
    let (theory_set, definition_types) = theories_with(
        r#"
        (def program diagonal-view :
          ([n.] ([.n] ix val))
          ->
          ([n.] (u64 val))
        = ([i.] u64.one))

        (def program reduce-diagonal :
          ([n.] {({[.n] u64} :) ({[.n] u64} :)})
          ->
          ([n.] (u64 val))
        = ([producer-len reduce-len.]
          {
            const.u64.0x0000000000000000
            (name.u64.add lift)
            (([.producer-len] :.param) name.diagonal-view lift)
            [.reduce-len]
          }
          reduce
        ))
        "#,
    );
    let program = TheoryId(op("program"));

    convert_theory(&theory_set, &definition_types, &program)
        .expect("generated copy arrows with n-dependent types should share the ambient context");
}

#[test]
fn theory_conversion_generates_diagonal_view_closure_with_shared_context() {
    let (theory_set, definition_types) = theories_with(
        r#"
        (def program diagonal-view :
          ([n.] ([.n] ix val))
          ->
          ([n.] (u64 val))
        = ([i.] u64.one))

        (def program reduce-diagonal :
          ([n.] {({[.n] u64} :) ({[.n] u64} :)})
          ->
          ([n.] (u64 val))
        = ([producer-len reduce-len.]
          {
            const.u64.0x0000000000000000
            (name.u64.add lift)
            (([.producer-len] :.param) name.diagonal-view lift)
            [.reduce-len]
          }
          reduce
        ))
        "#,
    );
    let program = TheoryId(op("program"));

    let converted =
        convert_theory(&theory_set, &definition_types, &program).expect("theory should convert");
    let Theory::Theory { arrows, .. } = converted else {
        panic!("program should be a theory");
    };

    let reduce_diagonal = arrows
        .get(&op("reduce-diagonal"))
        .expect("converted original definition should exist");
    let reduce_diagonal_body = reduce_diagonal
        .definition
        .as_ref()
        .expect("converted original definition should have a body");

    assert_operation_count(reduce_diagonal_body, "reducec", 1);
    assert_operation_count(reduce_diagonal_body, "reduce", 0);

    let closure_names = arrows
        .keys()
        .filter(|operation| operation.as_str().starts_with("closure.reduce-diagonal."))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(closure_names.len(), 2);

    // `diagonal-view` has type val(ix n) -> val(u64). Closure conversion
    // lowers the lifted producer closure to a generated arrow with source
    // 1, val(ix n) and target val(u64). The target does not mention `n`, but
    // both generated type maps must still share reduce-diagonal's ambient
    // context so positional variables keep the same meaning.
    let (closure_name, closure) = closure_names
        .iter()
        .filter_map(|name| arrows.get(name).map(|closure| (name, closure)))
        .find(|(_, closure)| closure.raw.type_maps.0.to_string().contains("ix"))
        .expect("generated diagonal-view closure should mention ix on its source type map");

    assert!(
        closure.definition.is_some(),
        "generated closure arrow {closure_name} should have a definition"
    );
    assert_eq!(
        closure.type_maps.0.sources, closure.type_maps.1.sources,
        "generated diagonal-view closure should use one shared context"
    );
    assert_eq!(
        closure.type_maps.0.sources.len(),
        1,
        "generated diagonal-view closure should inherit the ambient n context"
    );
    assert_eq!(
        closure.type_maps.0.targets.len(),
        2,
        "generated diagonal-view closure source should be 1, val(ix n)"
    );
    assert_eq!(
        closure.type_maps.1.targets.len(),
        1,
        "generated diagonal-view closure target should be val(u64)"
    );
    assert!(
        closure.raw.type_maps.1.to_string().contains("u64"),
        "target type map should contain val(u64)"
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

fn closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
    definition
        .hypergraph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, object)| is_closure_type(object).then_some(NodeId(index)))
        .collect()
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == FN_HOM_TYPE && children.len() == 2
}

fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn assert_same_definition_interface(actual: &AnnotatedTerm, expected: &AnnotatedTerm) {
    assert_converted_definition(
        actual,
        expected.hypergraph.edges.len(),
        interface_types(expected, &expected.sources),
        interface_types(expected, &expected.targets),
    );
}

fn assert_converted_definition(
    definition: &AnnotatedTerm,
    edges: usize,
    source_types: Vec<Obj>,
    target_types: Vec<Obj>,
) {
    assert_eq!(definition.hypergraph.edges.len(), edges);
    assert_eq!(
        interface_types(definition, &definition.sources),
        source_types
    );
    assert_eq!(
        interface_types(definition, &definition.targets),
        target_types
    );
}

fn assert_operation_count(term: &metacat::theory::Term, operation: &str, expected: usize) {
    let actual = term
        .hypergraph
        .edges
        .iter()
        .filter(|actual| actual.as_str() == operation)
        .count();
    assert_eq!(actual, expected, "operation count for `{operation}`");
}

fn obj(name: &str, children: Vec<Obj>) -> Obj {
    Tree::Node(op(name), 0, children)
}

fn binary_product(left: Obj, right: Obj) -> Obj {
    obj(PRODUCT_TYPE, vec![left, right])
}

fn right_associated_product(objects: &[Obj]) -> Obj {
    match objects {
        [] => obj(UNIT_TYPE, vec![]),
        [only] => only.clone(),
        [head, tail @ ..] => binary_product(head.clone(), right_associated_product(tail)),
    }
}

fn function_pointer_type(sources: Vec<Obj>, targets: Vec<Obj>) -> Obj {
    obj(
        "val",
        vec![obj("->", vec![pack_object(sources), pack_object(targets)])],
    )
}

fn pack_object(objects: Vec<Obj>) -> Obj {
    match objects.as_slice() {
        [] => obj("1", vec![]),
        [only] => only.clone(),
        _ => obj("*", objects),
    }
}

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
