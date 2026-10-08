use hexpr::{Hexpr, Operation};
use metacat::theory::{
    RawTheorySet,
    ast::{RawTheory, RawTheoryArrow},
};
use std::collections::BTreeSet;

use crate::{
    elaborate::ElaborateError,
    prefixes::{CONST_U32_PREFIX, CONST_U64_PREFIX},
};

#[derive(Debug, Clone, Copy)]
pub struct ConstantKind {
    prefix: &'static str,
    type_name: &'static str,
    hex_nibbles: usize,
}

pub const U64: ConstantKind = ConstantKind {
    prefix: CONST_U64_PREFIX,
    type_name: "u64",
    hex_nibbles: 16,
};

pub const U32: ConstantKind = ConstantKind {
    prefix: CONST_U32_PREFIX,
    type_name: "u32",
    hex_nibbles: 8,
};

/// For each operation `const.<type>.0x{c}` appearing in the source,
/// elaborate the theory with a constant `const.<type>.0x{c} : [] -> (<type> val)`.
pub fn elaborate(raw: &mut RawTheorySet, kind: ConstantKind) -> Result<(), ElaborateError> {
    let theory_names = raw.theories.keys().cloned().collect::<Vec<_>>();
    for theory_name in theory_names {
        let Some(theory) = raw.theories.get_mut(&theory_name) else {
            continue;
        };
        elaborate_theory(theory, kind)?;
    }
    Ok(())
}

/// Named literals in dependent interfaces denote nullary operations in the
/// syntax theory, even when generated source omitted their declarations.
pub(super) fn elaborate_type_literals(
    raw: &mut RawTheorySet,
    kind: ConstantKind,
) -> Result<(), ElaborateError> {
    let mut literals = BTreeSet::new();
    let prefix = format!("type.{}", kind.prefix);
    fn collect(expr: &Hexpr, prefix: &str, literals: &mut BTreeSet<Operation>) {
        match expr {
            Hexpr::Composition(parts) | Hexpr::Tensor(parts) => {
                for part in parts {
                    collect(part, prefix, literals);
                }
            }
            Hexpr::Frobenius { sources, targets } => {
                for variable in sources.iter().chain(targets) {
                    if let Some(label) = &variable.label {
                        collect(label, prefix, literals);
                    }
                }
            }
            Hexpr::Wire(inner) => collect(inner, prefix, literals),
            Hexpr::Operation(op) if op.as_str().starts_with(prefix) => {
                literals.insert(op.clone());
            }
            Hexpr::Hole | Hexpr::Operation(_) => {}
        }
    }
    for theory in raw
        .theories
        .values()
        .filter(|theory| theory.syntax_category.as_str() == "type")
    {
        for arrow in theory.arrows.values() {
            collect(&arrow.type_maps.0, &prefix, &mut literals);
            collect(&arrow.type_maps.1, &prefix, &mut literals);
        }
    }
    if literals.is_empty() {
        return Ok(());
    }
    let theory = raw
        .theories
        .get_mut(&"type".parse().expect("internal theory name"))
        .ok_or_else(|| ElaborateError::MissingTheory("type".into()))?;
    for literal in literals {
        let constant = literal
            .as_str()
            .strip_prefix("type.")
            .expect("type literal prefix")
            .parse()
            .expect("constant operation name");
        validate_constant(&constant, kind)?;
        let expected = RawTheoryArrow {
            name: literal.clone(),
            type_maps: (op("0"), op("1")),
            definition: None,
        };
        if let Some(existing) = theory.arrows.get(&literal) {
            if existing.definition.is_some() || existing.type_maps != expected.type_maps {
                return Err(ElaborateError::InvalidConstant {
                    operation: literal.to_string(),
                    reason: "type literal must be a primitive with interface 0 -> 1".into(),
                });
            }
        } else {
            theory.arrows.insert(literal, expected);
        }
    }
    Ok(())
}

fn elaborate_theory(theory: &mut RawTheory, kind: ConstantKind) -> Result<(), ElaborateError> {
    let constants = theory
        .arrows
        .values()
        .filter_map(|arrow| arrow.definition.as_ref())
        .flat_map(|definition| constants_in_hexpr(definition, kind))
        .collect::<Vec<_>>();

    for constant in constants {
        validate_constant(&constant, kind)?;
        theory
            .arrows
            .entry(constant.clone())
            .or_insert_with(|| const_arrow(constant, kind));
    }

    Ok(())
}

fn constants_in_hexpr(hexpr: &Hexpr, kind: ConstantKind) -> Vec<Operation> {
    let mut constants = Vec::new();
    collect_constants(hexpr, kind, &mut constants);
    constants
}

fn collect_constants(hexpr: &Hexpr, kind: ConstantKind, constants: &mut Vec<Operation>) {
    match hexpr {
        Hexpr::Composition(exprs) | Hexpr::Tensor(exprs) => {
            for expr in exprs {
                collect_constants(expr, kind, constants);
            }
        }
        Hexpr::Frobenius { .. } | Hexpr::Hole | Hexpr::Wire(_) => {}
        Hexpr::Operation(op) if op.as_str().starts_with(kind.prefix) => {
            constants.push(op.clone());
        }
        Hexpr::Operation(_) => {}
    }
}

fn validate_constant(op: &Operation, kind: ConstantKind) -> Result<(), ElaborateError> {
    let literal =
        op.as_str()
            .strip_prefix(kind.prefix)
            .ok_or_else(|| ElaborateError::InvalidConstant {
                operation: op.to_string(),
                reason: format!("expected prefix `{}`", kind.prefix),
            })?;
    let Some(hex) = literal.strip_prefix("0x") else {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: "expected a hexadecimal literal beginning with `0x`".to_string(),
        });
    };
    let hex = hex.replace('_', "");
    if hex.len() != kind.hex_nibbles {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: format!("expected exactly {} hexadecimal nibbles", kind.hex_nibbles),
        });
    }
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ElaborateError::InvalidConstant {
            operation: op.to_string(),
            reason: "literal contains a non-hexadecimal digit".to_string(),
        });
    }
    Ok(())
}

fn const_arrow(name: Operation, kind: ConstantKind) -> RawTheoryArrow {
    RawTheoryArrow {
        name,
        type_maps: (
            Hexpr::Frobenius {
                sources: Vec::new(),
                targets: Vec::new(),
            },
            Hexpr::Composition(vec![op(kind.type_name), op("val")]),
        ),
        definition: None,
    }
}

fn op(name: &str) -> Hexpr {
    Hexpr::Operation(name.parse().expect("generated operation should parse"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(type_declarations: &str, literal: &str) -> RawTheorySet {
        RawTheorySet::from_text(&format!(
            "(theory type nat {{{type_declarations}}})
             (theory program type {{
                (arr literal : ([] {{}}) -> ([] {{({{{literal} u64}} :)}}))
             }})"
        ))
        .unwrap()
    }

    #[test]
    fn interface_literals_get_missing_type_declarations() {
        for (kind, literal) in [
            (U64, "type.const.u64.0x0000000000000080"),
            (U32, "type.const.u32.0x00000080"),
        ] {
            let mut theories = raw("", literal);
            elaborate_type_literals(&mut theories, kind).unwrap();
            let theory = &theories.theories[&"type".parse().unwrap()];
            let arrow = &theory.arrows[&literal.parse().unwrap()];
            assert_eq!(arrow.type_maps, (op("0"), op("1")));
            assert!(arrow.definition.is_none());
            elaborate_type_literals(&mut theories, kind).unwrap();
            assert_eq!(theories.theories[&"type".parse().unwrap()].arrows.len(), 1);
        }
    }

    #[test]
    fn explicit_type_literals_must_have_the_expected_interface() {
        let literal = "type.const.u64.0x0000000000000080";
        for declaration in [
            format!("(arr {literal} : 0 -> 1)"),
            format!("(arr {literal} : 1 -> 1)"),
            format!("(def {literal} : 0 -> 1 = 1)"),
        ] {
            let mut theories = raw(&declaration, literal);
            assert_eq!(
                elaborate_type_literals(&mut theories, U64).is_ok(),
                declaration.starts_with("(arr") && declaration.contains(": 0 -> 1")
            );
        }
    }

    #[test]
    fn malformed_type_literals_are_rejected() {
        for literal in ["type.const.u64.0x80", "type.const.u64.0x00000000000000zz"] {
            assert!(matches!(
                elaborate_type_literals(&mut raw("", literal), U64),
                Err(ElaborateError::InvalidConstant { .. })
            ));
        }
    }
}
