//! Merge independently generated sources that share primitive declarations.
use std::collections::HashMap;

use hexpr::{Hexpr, LVar, Variable};
use metacat::theory::{RawTheorySet, ast::MergeRawError};

pub(crate) fn merge(
    existing: RawTheorySet,
    mut incoming: RawTheorySet,
) -> Result<RawTheorySet, MergeRawError> {
    for (name, theory) in &mut incoming.theories {
        let Some(previous) = existing.theories.get(name) else {
            continue;
        };
        theory.arrows.retain(|name, arrow| {
            let Some(other) = previous.arrows.get(name) else {
                return true;
            };
            // Keep definitions and incompatible interfaces for the strict merger
            // to reject. Only repeated primitive declarations can be omitted.
            !(arrow.definition.is_none()
                && other.definition.is_none()
                && normalized(&arrow.type_maps) == normalized(&other.type_maps))
        });
    }
    existing.merge(incoming)
}

fn normalized(maps: &(Hexpr, Hexpr)) -> (Hexpr, Hexpr) {
    fn variable(value: &mut LVar, names: &mut HashMap<Variable, Variable>) {
        let next = names.len();
        value.name = names
            .entry(value.name.clone())
            .or_insert_with(|| format!("v{next}").parse().expect("internal variable name"))
            .clone();
        if let Some(label) = &mut value.label {
            visit(label, names);
        }
    }
    fn visit(expr: &mut Hexpr, names: &mut HashMap<Variable, Variable>) {
        match expr {
            Hexpr::Composition(parts) | Hexpr::Tensor(parts) => {
                for part in parts {
                    visit(part, names);
                }
            }
            Hexpr::Frobenius { sources, targets } => {
                for value in sources.iter_mut().chain(targets) {
                    variable(value, names);
                }
            }
            Hexpr::Wire(inner) => visit(inner, names),
            Hexpr::Hole | Hexpr::Operation(_) => {}
        }
    }
    let mut maps = maps.clone();
    // Share the renaming across both maps to preserve dependent relationships.
    let mut names = HashMap::new();
    visit(&mut maps.0, &mut names);
    visit(&mut maps.1, &mut names);
    maps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(declaration: &str) -> RawTheorySet {
        RawTheorySet::from_text(&format!("(theory program type {{{declaration}}})")).unwrap()
    }

    #[test]
    fn identical_and_renamed_primitive_interfaces_are_accepted() {
        let first = "(arr identity : ([a . ] {([ . a] val)}) -> ([a . ] {([ . a] val)}))";
        for second in [
            first,
            "(arr identity : ([b . ] {([ . b] val)}) -> ([b . ] {([ . b] val)}))",
        ] {
            let merged = merge(raw(first), raw(second)).unwrap();
            assert_eq!(merged.theories[&"program".parse().unwrap()].arrows.len(), 1);
        }
    }

    #[test]
    fn different_inputs_outputs_and_dependencies_are_rejected() {
        let first = "(arr f : ([a b . ] {([ . a] val)}) -> ([a b . ] {([ . a] val)}))";
        for second in [
            "(arr f : ([a b . ] {([ . b] val)}) -> ([a b . ] {([ . a] val)}))",
            "(arr f : ([a b . ] {([ . a] val)}) -> ([a b . ] {([ . b] val)}))",
            "(arr f : ([a b . ] {(u64 val)}) -> ([a b . ] {([ . a] val)}))",
        ] {
            assert!(merge(raw(first), raw(second)).is_err());
        }
    }

    #[test]
    fn definitions_are_never_deduplicated() {
        let definition = "(def f : ([] {}) -> ([] {}) = [])";
        let primitive = "(arr f : ([] {}) -> ([] {}))";
        for (first, second) in [
            (definition, definition),
            (primitive, definition),
            (definition, primitive),
        ] {
            assert!(merge(raw(first), raw(second)).is_err());
        }
    }

    #[test]
    fn syntax_category_conflicts_are_rejected() {
        let first = raw("(arr f : ([] {}) -> ([] {}))");
        let second =
            RawTheorySet::from_text("(theory program nat {(arr f : ([] {}) -> ([] {}))})").unwrap();
        assert!(matches!(
            merge(first, second),
            Err(MergeRawError::SyntaxMismatch { .. })
        ));
    }

    #[test]
    fn shared_literals_are_accepted_in_both_theories() {
        let source = "
            (theory program type {
                (arr type.const.u64.0x0000000000000001
                    : ([] {}) -> ([] {({type.const.u64.0x0000000000000001 u64} :)}))
            })
            (theory type nat {(arr type.const.u64.0x0000000000000001 : 0 -> 1)})";
        let merged = merge(
            RawTheorySet::from_text(source).unwrap(),
            RawTheorySet::from_text(source).unwrap(),
        )
        .unwrap();
        for theory in merged.theories.values() {
            assert_eq!(theory.arrows.len(), 1);
        }
    }
}
