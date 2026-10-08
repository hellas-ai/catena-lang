//! Experimental callback primitives use compiler-generated runtime signatures.
//!
//! The exported stdlib declares callbacks as single closure inputs (`A => B`).
//! Replacing a closure exposes its captured environment and a function pointer
//! as two inputs, with the function taking `(environment, argument)`. For
//! example, `core.if` grows from five inputs to seven after converting both
//! branches. A generated closure body containing that call must therefore use
//! a converted declaration; checking it against the source signature fails.
//!
//! Even closed callbacks retain an environment input in this graph: unit `1`
//! (or a product of units) contains no runtime data, but still occupies a graph
//! port before erasure. Passing this environment does not allocate memory.
//! Partial applications can capture actual values, so its type remains generic.
//!
//! Legacy stdlib primitives already declare their converted interfaces, such
//! as `bool.ifc`. Keep those declarations and replacement rules separate;
//! synthesize signatures only for primitives listed by the experimental backend.
use super::{Obj, ReplaceClosuresError, boundary_to_hexpr, interface_types, interpret_type_maps};
use hexpr::Operation;
use metacat::{
    theory::{Theory, TheoryArrow, TheorySet, ast::RawTheoryArrow},
    tree::Tree,
};

use crate::{
    closure::region_conversion::RegionConversion,
    codegen::experimental::primitives::CONVERTED_PRIMITIVES, hexpr::term_to_hexpr,
    pass::forget_closures::ClosureForgotten,
};

/// Temporary post-conversion patch for the experimental backend. Region
/// discovery and replacement use the unchanged default conversion loop.
/// Once every callback has expanded, fix only the experimental operation names
/// and generated declarations before the completed theory is checked.
pub(in crate::closure) fn patch(
    conversion: &mut RegionConversion,
) -> Result<(), ReplaceClosuresError> {
    conversion.theory = declare_primitives(&conversion.theory)?;
    for (theory_id, definitions) in &mut conversion.generated_functions {
        let Theory::Theory { arrows, .. } = conversion
            .theory
            .theories
            .get_mut(theory_id)
            .expect("generated theory exists")
        else {
            unreachable!()
        };
        for (name, body) in definitions {
            for operation in &mut body.hypergraph.edges {
                rewrite_operation(operation);
            }
            let arrow = arrows.get_mut(name).expect("generated definition exists");
            arrow.definition = Some(body.clone().map_nodes(|_| ()));
            arrow.raw.definition = Some(term_to_hexpr(body));
        }
    }
    for definitions in conversion.terms.values_mut() {
        for body in definitions.values_mut() {
            for operation in &mut body.hypergraph.edges {
                if let ClosureForgotten::Operation(operation) = operation {
                    rewrite_operation(operation);
                }
            }
        }
    }
    Ok(())
}

fn rewrite_operation(operation: &mut Operation) {
    if let Some((_, converted)) = CONVERTED_PRIMITIVES
        .iter()
        .find(|(source, _)| operation.as_str() == *source)
    {
        *operation = converted
            .parse()
            .expect("converted primitive name should parse");
    }
}

/// Preserve source declarations while adding the runtime interfaces used by
/// generated bodies. Each A => B input becomes E, val((E * A) -> B), with
/// an independent environment type parameter, including when E is unit.
pub(crate) fn declare_primitives(
    theory_set: &TheorySet,
) -> Result<TheorySet, ReplaceClosuresError> {
    use crate::stdlib::constants::{FN_HOM_TYPE, FN_REF_TYPE, PRODUCT_TYPE, VALUE_TYPE};

    let mut output = theory_set.clone();
    for theory in output.theories.values_mut() {
        let Theory::Theory { syntax, arrows } = theory else {
            continue;
        };
        let syntax = &theory_set.theories[syntax];
        for &(source, converted) in CONVERTED_PRIMITIVES {
            let converted: Operation = converted.parse().expect("primitive name should parse");
            if arrows.contains_key(&converted) {
                continue;
            }
            let Some(original) = arrows.get(&source.parse().expect("primitive name should parse"))
            else {
                continue;
            };
            let mut context_arity = original.type_maps.0.sources.len();
            let mut sources = Vec::new();
            for object in interface_types(&original.type_maps.0)? {
                if let Tree::Node(operation, _, children) = &object
                    && operation.as_str() == FN_HOM_TYPE
                    && let [domain, codomain] = children.as_slice()
                {
                    // Each callback can capture a different environment. Use
                    // a fresh type parameter and share it between its input
                    // port and the first component of the function's domain.
                    let environment = Tree::Leaf(context_arity, ());
                    context_arity += 1;
                    let domain = Tree::Node(
                        PRODUCT_TYPE.parse().unwrap(),
                        0,
                        vec![environment.clone(), domain.clone()],
                    );
                    let function = Tree::Node(
                        FN_REF_TYPE.parse().unwrap(),
                        0,
                        vec![domain, codomain.clone()],
                    );
                    sources.push(environment);
                    sources.push(Tree::Node(VALUE_TYPE.parse().unwrap(), 0, vec![function]));
                } else {
                    sources.push(object);
                }
            }
            let targets = interface_types(&original.type_maps.1)?;
            let context = (0..context_arity).map(|index| (index, index)).collect();
            let raw = RawTheoryArrow {
                name: converted.clone(),
                type_maps: (
                    boundary_to_hexpr(&sources, &context)?,
                    boundary_to_hexpr(&targets, &context)?,
                ),
                definition: None,
            };
            let type_maps = interpret_type_maps(syntax, &raw.type_maps)?;
            arrows.insert(
                converted.clone(),
                TheoryArrow {
                    name: converted,
                    raw,
                    type_maps,
                    definition: None,
                },
            );
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metacat::theory::TheoryId;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn experimental_declarations_do_not_synthesize_legacy_primitives() {
        let raw = metacat::theory::RawTheorySet::from_texts(crate::stdlib::sources()).unwrap();
        let mut theory = TheorySet::from_raw(crate::elaborate::elaborate(raw).unwrap()).unwrap();
        let program = TheoryId("program".parse().unwrap());
        let Theory::Theory { arrows, .. } = theory.theories.get_mut(&program).unwrap() else {
            unreachable!()
        };
        // A missing legacy runtime declaration must stay missing rather than
        // silently acquiring a signature synthesized by the experimental path.
        arrows.remove(&"bool.ifc".parse().unwrap());
        let names = arrows.keys().cloned().collect::<BTreeSet<_>>();
        let output = declare_primitives(&theory).unwrap();
        let Theory::Theory { arrows, .. } = &output.theories[&program] else {
            unreachable!()
        };
        assert_eq!(arrows.keys().cloned().collect::<BTreeSet<_>>(), names);
        assert_eq!(
            crate::codegen::experimental::primitives::source_primitive("bool.ifc"),
            "bool.ifc"
        );
        assert!(
            !super::super::CONVERTED_PRIMITIVES
                .iter()
                .any(|(source, _)| *source == "core.if")
        );
        assert!(
            !CONVERTED_PRIMITIVES
                .iter()
                .any(|(source, _)| *source == "bool.if")
        );
    }

    #[test]
    fn nested_experimental_primitive_checks_with_converted_signature() {
        let source = r#"
        (theory program type {
        (arr core.if
          : ([a b .] {(bool val) [.a] ({[.a] [.b]} =>) ({[.a] [.b]} =>) 1})
          -> ([a b .] {[.b]}))
        (def choose
          : {({(bool val) (bool val)} =>) ({(bool val) (bool val)} =>)
             (bool val) (bool val)}
          -> (bool val)
          = ([left right flag value .]
              {[.flag value left right] unit.intro} core.if))
        (def nested
          : {(bool val) (bool val)} -> (bool val)
          = ([flag value .]
              {[.flag value]
               ({(name.bool.id lift) (name.bool.not lift) [.flag]} partial.choose.3)
               (name.bool.id lift) unit.intro}
              core.if))
        })
        "#;
        let raw =
            metacat::theory::RawTheorySet::from_texts(crate::stdlib::sources().chain([source]))
                .unwrap();
        let mut theory = TheorySet::from_raw(crate::elaborate::elaborate(raw).unwrap()).unwrap();
        let Theory::Theory { arrows, .. } = theory
            .theories
            .get_mut(&TheoryId("program".parse().unwrap()))
            .unwrap()
        else {
            unreachable!()
        };
        for (name, arrow) in arrows {
            // Default library bodies belong to default conversion. Keep only
            // this experimental fixture and its synthesized partial helpers.
            if name.as_str() != "nested" && !name.as_str().contains("choose") {
                arrow.definition = None;
            }
        }
        let theory = crate::pass::inline_definitions::run(
            &theory,
            &BTreeMap::from([(
                TheoryId("program".parse().unwrap()),
                BTreeSet::from([
                    "choose".parse().unwrap(),
                    "partial.choose.3".parse().unwrap(),
                ]),
            )]),
        )
        .unwrap();
        let types = crate::check::check(&theory).unwrap();
        let forgotten = crate::pass::forget_closures::run(&theory, &types).unwrap();
        let conversion = crate::closure::run_with_progress(
            &theory,
            &forgotten,
            crate::codegen::CodegenKind::Experimental,
            &mut |_| {},
        )
        .unwrap();
        let program = &conversion.generated_theory.theories[&TheoryId("program".parse().unwrap())];
        assert_eq!(
            program
                .get_arrow(&"core.if".parse().unwrap())
                .unwrap()
                .type_maps
                .0
                .targets
                .len(),
            5
        );
        assert_eq!(
            program
                .get_arrow(&"core.ifc".parse().unwrap())
                .unwrap()
                .type_maps
                .0
                .targets
                .len(),
            7
        );
        assert!(
            conversion
                .generated_functions
                .values()
                .flat_map(|definitions| definitions.values())
                .any(|body| body
                    .hypergraph
                    .edges
                    .iter()
                    .any(|op| op.as_str() == "core.ifc"))
        );
        crate::check::check(&conversion.generated_theory).unwrap();
    }
}
