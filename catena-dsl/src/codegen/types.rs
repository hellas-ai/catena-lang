use hexpr::Operation;
use metacat::tree::Tree;

/// Map a Catena object type to the type string stored on a `StructuredProgram` parameter.
///
/// Example:
/// `bool` becomes `uint8_t`
///
/// Example:
/// output `bool` becomes `uint8_t*`
pub fn structured_param_type(obj: &Tree<(), Operation>, by_pointer: bool) -> Option<String> {
    let base = scalar_type(obj)?;
    if by_pointer {
        Some(format!("{base}*"))
    } else {
        Some(base)
    }
}

/// Render a Catena object type as a full C parameter declaration.
///
/// Example:
/// `bool` with name `x0` becomes `uint8_t x0`
///
/// Example:
/// output `bool` with name `out_x1` becomes `uint8_t *out_x1`
pub fn c_param_decl(obj: &Tree<(), Operation>, name: &str, by_pointer: bool) -> Option<String> {
    if by_pointer {
        scalar_type(obj).map(|ty| format!("{ty} *{name}"))
    } else {
        declaration(obj, name)
    }
}

/// Render a Catena object type as a full C local declaration.
///
/// Example:
/// `bool` with name `x3` becomes `uint8_t x3`
///
/// Example:
/// `(bool -> bool)` with name `f` becomes `uint8_t (*f)(uint8_t arg0)`
pub fn c_local_decl(obj: &Tree<(), Operation>, name: &str) -> Option<String> {
    declaration(obj, name)
}

fn scalar_type(obj: &Tree<(), Operation>) -> Option<String> {
    match obj {
        Tree::Node(op, 0, children) if op.as_str() == "bool" && children.is_empty() => {
            Some("uint8_t".to_string())
        }
        Tree::Node(op, 0, children) if op.as_str() == "->" => {
            let [source, target] = children.as_slice() else {
                return None;
            };
            Some(format!(
                "fn_{}_to_{}",
                mangle_object(source),
                mangle_object(target)
            ))
        }
        _ => None,
    }
}

fn declaration(obj: &Tree<(), Operation>, name: &str) -> Option<String> {
    match obj {
        Tree::Node(op, 0, children) if op.as_str() == "bool" && children.is_empty() => {
            Some(format!("uint8_t {name}"))
        }
        Tree::Node(op, 0, children) if op.as_str() == "->" => {
            let [source, target] = children.as_slice() else {
                return None;
            };
            let return_ty = scalar_type(target)?;
            let params = c_param_list(source)?;
            Some(format!("{return_ty} (*{name})({params})"))
        }
        _ => None,
    }
}

fn c_param_list(obj: &Tree<(), Operation>) -> Option<String> {
    let args = flatten_product(obj);
    if args.is_empty() {
        return Some("void".to_string());
    }
    args.into_iter()
        .enumerate()
        .map(|(index, arg)| declaration(arg, &format!("arg{index}")))
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.join(", "))
}

fn flatten_product<'a>(obj: &'a Tree<(), Operation>) -> Vec<&'a Tree<(), Operation>> {
    match obj {
        Tree::Node(op, 0, children) if op.as_str() == "*" => {
            children.iter().flat_map(flatten_product).collect()
        }
        Tree::Node(op, 0, children) if op.as_str() == "1" && children.is_empty() => Vec::new(),
        other => vec![other],
    }
}

fn mangle_object(obj: &Tree<(), Operation>) -> String {
    match obj {
        Tree::Empty => "empty".to_string(),
        Tree::Leaf(index, _) => format!("x{index}"),
        Tree::Node(op, port, children) => {
            let mut out = sanitize_ident(op.as_str());
            if *port != 0 {
                out.push('_');
                out.push_str(&port.to_string());
            }
            for child in children {
                out.push('_');
                out.push_str(&mangle_object(child));
            }
            out
        }
    }
}

fn sanitize_ident(value: &str) -> String {
    let mut ident = value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.is_empty() {
        ident.push('_');
    }
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_type() -> Tree<(), Operation> {
        Tree::Node("bool".parse().unwrap(), 0, vec![])
    }

    fn fn_ptr_type(source: Tree<(), Operation>, target: Tree<(), Operation>) -> Tree<(), Operation> {
        Tree::Node("->".parse().unwrap(), 0, vec![source, target])
    }

    #[test]
    fn structured_param_type_renders_function_pointer_name() {
        let ty = fn_ptr_type(bool_type(), bool_type());
        assert_eq!(
            structured_param_type(&ty, false).as_deref(),
            Some("fn_bool_to_bool")
        );
        assert_eq!(
            structured_param_type(&ty, true).as_deref(),
            Some("fn_bool_to_bool*")
        );
    }

    #[test]
    fn c_declarations_render_function_pointer_types() {
        let ty = fn_ptr_type(bool_type(), bool_type());
        assert_eq!(
            c_param_decl(&ty, "f", false).as_deref(),
            Some("uint8_t (*f)(uint8_t arg0)")
        );
        assert_eq!(
            c_local_decl(&ty, "tmp").as_deref(),
            Some("uint8_t (*tmp)(uint8_t arg0)")
        );
    }
}
