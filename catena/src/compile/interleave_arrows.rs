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
        BoundaryTensor {
            pack_with: "+".parse().expect("valid operation"),
            unit: "0".parse().expect("valid operation"),
            target_wire_shape: "*".parse().expect("valid operation"),
        },
    );
    let data_in_control = tensor_pack_embed(
        syntax,
        &data,
        &control,
        BoundaryTensor {
            pack_with: "*".parse().expect("valid operation"),
            unit: "1".parse().expect("valid operation"),
            target_wire_shape: "+".parse().expect("valid operation"),
        },
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
    boundary: BoundaryTensor,
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

            let source_map = prepare_boundary_type_map(&arrow.type_maps.0, syntax, &boundary);
            let target_map = prepare_boundary_type_map(&arrow.type_maps.1, syntax, &boundary);

            Some(RawTheoryArrow {
                name: lifted_name,
                type_maps: (source_map, target_map),
                definition: None,
            })
        })
        .collect()
}

#[derive(Clone)]
struct BoundaryTensor {
    pack_with: Operation,
    unit: Operation,
    target_wire_shape: Operation,
}

fn prepare_boundary_type_map(map: &Hexpr, syntax: &Theory, boundary: &BoundaryTensor) -> Hexpr {
    if let Some(exposed) = expose_boundary_tensor(map, &boundary.target_wire_shape) {
        exposed
    } else {
        pack_type_map(map, syntax, &boundary.pack_with, &boundary.unit)
    }
}

/// Expose an explicit boundary tensor when it already matches the target theory's wire shape.
///
/// For example, when lifting data into control, a data target like `({a b} +)` already describes
/// a control-style alternative boundary. We expose it as `{a b}` instead of packing the two wires
/// again into one control value.
///
/// This only exposes terms built from the target tensor itself. Other operations are treated as
/// opaque leaves, so `(({a b} +) value)` is left unchanged rather than rewriting under `value`.
fn expose_boundary_tensor(map: &Hexpr, target_wire_shape: &Operation) -> Option<Hexpr> {
    let Hexpr::Composition(steps) = map else {
        return None;
    };
    let Some(last) = steps.last() else {
        return None;
    };

    // If the final step already is the target theory's wire shape, remove it:
    // the target theory should see the individual wires, not a repacked object.
    if let Hexpr::Operation(op) = last
        && op == target_wire_shape
    {
        let exposed = match &steps[..steps.len() - 1] {
            [] => return None,
            [only] => only.clone(),
            rest => Hexpr::Composition(rest.to_vec()),
        };
        return Some(expose_nested_boundary_tensors(&exposed, target_wire_shape));
    }

    // A boundary tensor can be the final step of a larger contextual map, for
    // example `[i . i] ; ({a b} +)`. Recurse only through the final step so
    // earlier operations remain opaque.
    let exposed_last = expose_boundary_tensor(last, target_wire_shape)?;
    let mut exposed = steps[..steps.len() - 1].to_vec();
    exposed.push(exposed_last);
    match exposed.as_slice() {
        [only] => Some(only.clone()),
        _ => Some(Hexpr::Composition(exposed)),
    }
}

fn expose_nested_boundary_tensors(map: &Hexpr, target_wire_shape: &Operation) -> Hexpr {
    if let Some(exposed) = expose_boundary_tensor(map, target_wire_shape) {
        return exposed;
    }

    match map {
        Hexpr::Tensor(factors) => Hexpr::Tensor(
            factors
                .iter()
                .map(|factor| expose_nested_boundary_tensors(factor, target_wire_shape))
                .collect(),
        ),
        // Do not recurse through arbitrary operations. Only explicit target
        // tensor structure is exposed; all other constructors remain opaque.
        _ => map.clone(),
    }
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
            for first_pass_through in 2..n {
                let mut factors = vec![Hexpr::Operation(tensor.clone())];
                factors.extend((first_pass_through..n).map(identity_hexpr));
                steps.push(Hexpr::Tensor(factors));
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
    use metacat::theory::{RawTheorySet, TheoryId};

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

    #[test]
    fn interleaved_arrows_typecheck_for_large_tensor_packs() {
        let source = r#"
        (theory syntax nat {
          (arr * : 2 -> 1)
          (arr 1 : 0 -> 1)
          (arr + : 2 -> 1)
          (arr 0 : 0 -> 1)
          (arr f32 : 0 -> 1)
        })

        (theory data syntax {
          (arr wide : {f32 f32 f32 f32} -> {f32 f32 f32})
        })

        (theory control syntax {
          (arr id : f32 -> f32)

          # after interleaving, this should typecheck only if wide source and
          # target tensors are packed with the expected left-associated shape
          (def expected :
            ({({({f32 f32} *) f32} *) f32} *) ->
            ({({f32 f32} *) f32} *)
            =
            data.wide)
        })
        "#;

        let raw = RawTheorySet::from_text(source).unwrap();
        let elaborated = crate::check::elaborate(&raw).unwrap();
        let checked = crate::check::check(&elaborated).unwrap();
        assert!(
            checked
                .theories
                .get(&TheoryId("control".parse().unwrap()))
                .and_then(|theory| theory.get_arrow(&"data.wide".parse().unwrap()))
                .is_some()
        );
    }

    #[test]
    fn interleaved_arrows_expose_boundary_tensors_in_both_directions() {
        let source = r#"
        (theory syntax nat {
          (arr * : 2 -> 1)
          (arr 1 : 0 -> 1)
          (arr + : 2 -> 1)
          (arr 0 : 0 -> 1)
          (arr f32 : 0 -> 1)
        })

        (theory data syntax {
          (arr data-to-control-source : ({f32 f32} +) -> f32)
          (arr data-to-control-target : f32 -> ({f32 f32} +))

          # after interleaving, these should typecheck only if control's
          # product boundary is exposed when lifted into data
          (def expected-control-to-data-source :
            {f32 f32} -> f32
            =
            control.control-to-data-source)
          (def expected-control-to-data-target :
            f32 -> {f32 f32}
            =
            control.control-to-data-target)
        })

        (theory control syntax {
          (arr control-to-data-source : ({f32 f32} *) -> f32)
          (arr control-to-data-target : f32 -> ({f32 f32} *))

          # after interleaving, these should typecheck only if data's
          # coproduct boundary is exposed when lifted into control
          (def expected-data-to-control-source :
            {f32 f32} -> f32
            =
            data.data-to-control-source)
          (def expected-data-to-control-target :
            f32 -> {f32 f32}
            =
            data.data-to-control-target)
        })
        "#;

        let raw = RawTheorySet::from_text(source).unwrap();
        let elaborated = crate::check::elaborate(&raw).unwrap();
        let checked = crate::check::check(&elaborated).unwrap();
        assert!(
            checked
                .theories
                .get(&TheoryId("control".parse().unwrap()))
                .and_then(|theory| theory.get_arrow(&"data.data-to-control-target".parse().unwrap()))
                .is_some()
        );
        assert!(
            checked
                .theories
                .get(&TheoryId("data".parse().unwrap()))
                .and_then(
                    |theory| theory.get_arrow(&"control.control-to-data-target".parse().unwrap())
                )
                .is_some()
        );
    }

    #[test]
    fn interleaved_arrows_expose_boundary_tensors_in_longer_terms() {
        let source = r#"
        (theory syntax nat {
          (arr * : 2 -> 1)
          (arr 1 : 0 -> 1)
          (arr + : 2 -> 1)
          (arr 0 : 0 -> 1)
          (arr f32 : 0 -> 1)
          (arr value : 1 -> 1)
        })

        (theory data syntax {
          (arr data-to-control-long :
            ({f32 ({f32 f32} +)} +) ->
            ({({f32 f32} +) f32} +))
          (arr data-to-control-mixed :
            (({f32 f32} +) value) ->
            f32)

          # after interleaving, this should typecheck only if every product
          # boundary is exposed when lifted into data
          (def expected-control-to-data-long :
            {f32 {f32 f32}} ->
            {{f32 f32} f32}
            =
            control.control-to-data-long)
          (def expected-control-to-data-mixed :
            (({f32 f32} *) value) ->
            f32
            =
            control.control-to-data-mixed)
        })

        (theory control syntax {
          (arr control-to-data-long :
            ({f32 ({f32 f32} *)} *) ->
            ({({f32 f32} *) f32} *))
          (arr control-to-data-mixed :
            (({f32 f32} *) value) ->
            f32)

          # after interleaving, this should typecheck only if every coproduct
          # boundary is exposed when lifted into control
          (def expected-data-to-control-long :
            {f32 {f32 f32}} ->
            {{f32 f32} f32}
            =
            data.data-to-control-long)
          (def expected-data-to-control-mixed :
            (({f32 f32} +) value) ->
            f32
            =
            data.data-to-control-mixed)
        })
        "#;

        let raw = RawTheorySet::from_text(source).unwrap();
        let elaborated = crate::check::elaborate(&raw).unwrap();
        crate::check::check(&elaborated).unwrap();
    }
}
