//! Mutually interleave the two "control" and "data" theories by adding "product-packed" data
//! arrows to control, and "coproduct-packed" control arrows to data.
//!
//! As a concrete example, the fragment of `stdlib.hex` would be elaborated with the "arr"
//! declarations commented.
//!
//! ```text
//! (theory syntax nat {
//!   # product/unit, coproduct/counit
//!   (arr * : 2 -> 1)
//!   (arr 1 : 0 -> 1)
//!   (arr + : 2 -> 1)
//!   (arr 0 : 0 -> 1)
//!
//!   # Builtin dtype definitions
//!   (arr f32 : 0 -> 1)
//!   (arr bool : 0 -> 1)
//! })
//!
//! # define the "data theory" (in this example, we just have one dataflow map)
//! (theory data syntax {
//!   (arr f32.add : {f32 f32} -> f32)
//! })
//!
//! # define the control theory - we'll assume no arrows added by the user in this minimal example
//! (theory control syntax {
//!   # elaboration adds a generating arrow to control: data.f32.add.
//!   # By convention, the compiler knows to interpret this as the data theory's "f32.add" arrow.
//!   (arr data.f32.add : ({f32 f32} *) -> f32)
//! })
//!
//! # conversely, if control had a map with multiple outputs, elaboration would add a
//! # coproduct-packed version to data.
//! (theory control syntax {
//!   (arr branch : flag -> {f32 f32})
//! })
//!
//! (theory data syntax {
//!   # elaboration adds a generating arrow to data: control.branch.
//!   # By convention, the compiler knows to interpret this as the control theory's "branch" arrow.
//!   (arr control.branch : flag -> ({f32 f32} +))
//! })
//! ```
use hexpr::{Hexpr, Operation};
use metacat::theory::{
    RawTheorySet, Theory,
    ast::{RawTheory, RawTheoryArrow},
};
use open_hypergraphs::category::Arrow;
use std::str::FromStr;

/// Interleave "control" maps into "data" and vice-versa.
// Sketch:
// - Interpret the syntax theory (so we can get type map coarities)
// - Compute declarations to add for "control-in-data"
// - Compute declarations to add for "data-in-control"
// - Add both sets to their corresponding theories
// We compute the arrows to add before adding so that *synthesized* arrows are not copied.
pub fn interleave(syntax: &Theory, raw: &mut RawTheorySet) {
    let control_name: Operation = "control".parse().expect("valid operation");
    let data_name: Operation = "data".parse().expect("valid operation");

    let Some(control) = raw.theories.get(&control_name).cloned() else {
        return;
    };
    let Some(data) = raw.theories.get(&data_name).cloned() else {
        return;
    };

    let control_in_data = tensor_pack_embed(
        syntax,
        &control,
        &data,
        "+".parse().expect("valid operation"),
        "0".parse().expect("valid operation"),
    );
    let data_in_control = tensor_pack_embed(
        syntax,
        &data,
        &control,
        "*".parse().expect("valid operation"),
        "1".parse().expect("valid operation"),
    );

    if let Some(data_theory) = raw.theories.get_mut(&data_name) {
        for arrow in control_in_data {
            data_theory.arrows.insert(arrow.name.clone(), arrow);
        }
    }
    if let Some(control_theory) = raw.theories.get_mut(&control_name) {
        for arrow in data_in_control {
            control_theory.arrows.insert(arrow.name.clone(), arrow);
        }
    }
}

/// Let C and D be symmetric monoidal categories over the same syntax category S.
/// This function creates a generating operation in D for each generating operation in C with
/// arity/coarity `1 -> 1`, whose type is the left-biased "tensor-packing" defined in `pack`.
///
/// Procedure: for each arrow (both declared arr and definitions def) in C,
///
///  - Compute a "packed" version of the type, so e.g., a type map like {f32 f32} becomes `({f32 f32} *)`
///  - Create an arr declaration (as syntax) with the packed type maps in dataflow
///
/// Returns a list of `TheoryArrow` declarations (not definitions) to add to D.
fn tensor_pack_embed(
    syntax: &Theory,
    source: &RawTheory,
    target: &RawTheory,
    tensor: Operation,
    unit: Operation,
) -> Vec<RawTheoryArrow> {
    source
        .arrows
        .values()
        .filter_map(|arrow| {
            let lifted_name: Operation = format!("{}.{}", source.name, arrow.name)
                .parse()
                .expect("lifted operation name should parse");
            if target.arrows.contains_key(&lifted_name) {
                return None;
            }

            Some(RawTheoryArrow {
                name: lifted_name,
                type_maps: (
                    pack_type_map(&arrow.type_maps.0, syntax, &tensor, &unit),
                    pack_type_map(&arrow.type_maps.1, syntax, &tensor, &unit),
                ),
                definition: None,
            })
        })
        .collect()
}

/// Let `A = A₀ × A₁ × ... × Am` be an object (list of generating objects) in a chosen syntax category.
/// Then `pack(A, ●, I)` computes a hexpr with type `A₀ × A₁ × ... × Am -> A₀ ● A₁ ● ... ● Am`,
/// where `●` is a chosen binary tensor product, and `I` its unit.
/// concretely, we have cases:
///
/// - `I` when `m = 0`
/// - `A` when `m = 1`
/// - `head(A) ● pack(tail(A))` when `m ≥ 2`
fn pack(object_size: usize, tensor: Operation, unit: Operation) -> Hexpr {
    match object_size {
        0 => Hexpr::Operation(unit),
        1 => identity_hexpr(0),
        n => {
            let mut steps = Vec::new();
            for next_var in 2..n {
                steps.push(Hexpr::Tensor(vec![
                    Hexpr::Operation(tensor.clone()),
                    identity_hexpr(next_var),
                ]));
            }
            steps.push(Hexpr::Operation(tensor));
            Hexpr::Composition(steps)
        }
    }
}

// Compute the composition of a type map with its corresponding 'pack' morphism
fn pack_type_map(map: &Hexpr, syntax: &Theory, tensor: &Operation, unit: &Operation) -> Hexpr {
    let interpreted = hexpr::try_interpret(&syntax.local_signature(), map)
        .expect("type map should interpret in the resolved syntax theory");
    match interpreted.target().len() {
        1 => map.clone(),
        n => Hexpr::Composition(vec![map.clone(), pack(n, tensor.clone(), unit.clone())]),
    }
}

fn identity_hexpr(var_index: usize) -> Hexpr {
    let name = format!("x{var_index}");
    let var = hexpr::Variable::from_str(&name).expect("generated variable should parse");
    Hexpr::Frobenius {
        sources: vec![var.clone()],
        targets: vec![var],
    }
}

#[cfg(test)]
mod tests {
    use metacat::theory::RawTheorySet;

    #[test]
    fn interleaved_arrows_typecheck_in_both_directions() {
        let source = r#"
        (theory syntax nat {
          (arr * : 2 -> 1)
          (arr 1 : 0 -> 1)
          (arr + : 2 -> 1)
          (arr 0 : 0 -> 1)
          (arr f32 : 0 -> 1)
        })

        (theory data syntax {
          (arr f32.add : {f32 f32} -> f32)

          # after interleaving, this should typecheck
          (def merge : ({1 1} +) -> 1 = control.merge)

        })

        (theory control syntax {
            (arr merge : ({1 1} +) -> 1)

            # after interleaving, this should typecheck
            (def expected : ({f32 f32} *) -> f32 = data.f32.add)
        })
        "#;

        let raw = RawTheorySet::from_text(source).unwrap();
        let elaborated = crate::check::elaborate(&raw).unwrap();
        crate::check::check(&elaborated).unwrap();
    }
}
