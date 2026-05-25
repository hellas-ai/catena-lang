//! Elaborate a theory by adding a symbol `name.f : I -> (A -> B)` for each arrow `f : A -> B`.
//! This follows from "finitary closed monoidal categories".
use hexpr::{Hexpr, Operation, Variable, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};

use crate::elaborate::ElaborateError;

const FN_TYPE: &str = "->";
const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const NAME_PREFIX: &str = "name.";

pub fn elaborate_theory(
    raw: &mut RawTheorySet,
    theory_name: &Operation,
) -> Result<(), ElaborateError> {
    let theory = raw
        .theories
        .get(theory_name)
        .expect("requested theory should exist");

    let syntax_theory_name = theory.syntax_category.clone();
    let raw_syntax_dependencies = transitive_dependency_subset([syntax_theory_name.clone()], raw)?;
    let syntax_dependencies = TheorySet::from_raw(raw_syntax_dependencies)?;
    let syntax = syntax_dependencies
        .theories
        .get(&TheoryId(syntax_theory_name))
        .expect("interpreted syntax theory should exist");

    let theory = raw
        .theories
        .get_mut(theory_name)
        .expect("requested theory should exist");
    elaborate_theory_with_interpreted_syntax(theory, syntax);
    Ok(())
}

fn elaborate_theory_with_interpreted_syntax(raw: &mut RawTheory, syntax: &Theory) {
    let mut new_arrows = Vec::new();
    for arrow in raw.arrows.values() {
        new_arrows.push(name_arrow(syntax, arrow));
    }

    for arrow in new_arrows {
        raw.arrows.insert(arrow.name.clone(), arrow);
    }
}

fn name_arrow(syntax: &Theory, raw: &RawTheoryArrow) -> RawTheoryArrow {
    RawTheoryArrow {
        name: format!("{NAME_PREFIX}{}", raw.name)
            .parse()
            .expect("generated operation name should satisfy hexpr operation syntax"),
        type_maps: (source_type_map(syntax, raw), target_type_map(syntax, raw)),
        definition: None,
    }
}

fn source_type_map(syntax: &Theory, raw: &RawTheoryArrow) -> Hexpr {
    let interpreted_source = try_interpret(&syntax.local_signature(), &raw.type_maps.0)
        .expect("raw source type map should interpret in the provided syntax theory");
    let metavars = vars("x", interpreted_source.sources.len());

    Hexpr::Frobenius {
        sources: metavars.clone(),
        targets: metavars,
    }
}

fn target_type_map(syntax: &Theory, raw: &RawTheoryArrow) -> Hexpr {
    let interpreted_source = try_interpret(&syntax.local_signature(), &raw.type_maps.0)
        .expect("raw source type map should interpret in the provided syntax theory");
    let interpreted_target = try_interpret(&syntax.local_signature(), &raw.type_maps.1)
        .expect("raw target type map should interpret in the provided syntax theory");

    let metavars = vars("x", interpreted_source.sources.len());
    let mut copied_metavars = metavars.clone();
    copied_metavars.extend(metavars.clone());
    let copy = Hexpr::Frobenius {
        sources: metavars,
        targets: copied_metavars,
    };

    let pack_s = Hexpr::Composition(vec![
        raw.type_maps.0.clone(),
        pack_object(interpreted_source.targets.len()),
    ]);
    let pack_t = Hexpr::Composition(vec![
        raw.type_maps.1.clone(),
        pack_object(interpreted_target.targets.len()),
    ]);

    Hexpr::Composition(vec![copy, Hexpr::Tensor(vec![pack_s, pack_t]), op(FN_TYPE)])
}

fn pack_object(object_size: usize) -> Hexpr {
    match object_size {
        0 => op(UNIT_TYPE),
        1 => identity_var("x0"),
        2 => op(PRODUCT_TYPE),
        n => Hexpr::Composition(vec![
            Hexpr::Tensor(vec![
                pack_object(n - 1),
                identity_var(&format!("x{}", n - 1)),
            ]),
            op(PRODUCT_TYPE),
        ]),
    }
}

fn vars(prefix: &str, arity: usize) -> Vec<Variable> {
    (0..arity).map(|i| var(&format!("{prefix}{i}"))).collect()
}

fn var(name: &str) -> Variable {
    name.parse()
        .expect("generated variable should satisfy hexpr variable syntax")
}

fn identity_var(name: &str) -> Hexpr {
    Hexpr::Frobenius {
        sources: vec![var(name)],
        targets: vec![var(name)],
    }
}

fn op(name: &str) -> Hexpr {
    Hexpr::Operation(
        name.parse::<Operation>()
            .expect("generated operation should satisfy hexpr operation syntax"),
    )
}
