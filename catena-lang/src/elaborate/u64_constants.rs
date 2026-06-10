use hexpr::{Hexpr, Operation};
use metacat::theory::{
    RawTheorySet,
    ast::{RawTheory, RawTheoryArrow},
};

use crate::elaborate::ElaborateError;

const U64_CONST_PREFIX: &str = "const.u64.";

/// For each operation `const.u64.0x{c}` appearing in the source,
/// elaborate the theory with a constant `const.u64.0x{c} : [] -> (u64 val)`
pub fn elaborate(raw: &mut RawTheorySet) -> Result<(), ElaborateError> {
    let theory_names = raw.theories.keys().cloned().collect::<Vec<_>>();
    for theory_name in theory_names {
        let Some(theory) = raw.theories.get_mut(&theory_name) else {
            continue;
        };
        elaborate_theory(theory)?;
    }
    Ok(())
}

fn elaborate_theory(theory: &mut RawTheory) -> Result<(), ElaborateError> {
    let constants = theory
        .arrows
        .values()
        .filter_map(|arrow| arrow.definition.as_ref())
        .flat_map(constants_in_hexpr)
        .collect::<Vec<_>>();

    for constant in constants {
        theory
            .arrows
            .entry(constant.clone())
            .or_insert_with(|| u64_const_arrow(constant));
    }

    Ok(())
}

fn constants_in_hexpr(hexpr: &Hexpr) -> Vec<Operation> {
    let mut constants = Vec::new();
    collect_constants(hexpr, &mut constants);
    constants
}

fn collect_constants(hexpr: &Hexpr, constants: &mut Vec<Operation>) {
    match hexpr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                collect_constants(expr, constants);
            }
        }
        Hexpr::Frobenius { .. } => {}
        Hexpr::Operation(op) if op.as_str().starts_with(U64_CONST_PREFIX) => {
            constants.push(op.clone());
        }
        Hexpr::Operation(_) => {}
    }
}

fn u64_const_arrow(name: Operation) -> RawTheoryArrow {
    RawTheoryArrow {
        name,
        type_maps: (
            Hexpr::Frobenius {
                sources: Vec::new(),
                targets: Vec::new(),
            },
            Hexpr::Composition(vec![op("u64"), op("val")]),
        ),
        definition: None,
    }
}

fn op(name: &str) -> Hexpr {
    Hexpr::Operation(name.parse().expect("generated operation should parse"))
}
