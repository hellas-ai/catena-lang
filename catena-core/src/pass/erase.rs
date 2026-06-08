//! Erase type-level operations

use crate::lang::{Arr, Obj};
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

#[derive(Clone)]
pub struct Erase {
    value: Option<&'static str>,
    recursive: bool,
}

impl Erase {
    pub fn default_value() -> Self {
        Self {
            value: None,
            recursive: false,
        }
    }

    pub fn with_value(value: &'static str) -> Self {
        Self {
            value: Some(value),
            recursive: true,
        }
    }

    pub fn with_value_shallow(value: &'static str) -> Self {
        Self {
            value: Some(value),
            recursive: false,
        }
    }

    fn is_value(&self, o: &Obj) -> bool {
        let value = self.value.unwrap_or("value");
        if self.recursive {
            contains_value(o, value)
        } else {
            is_value_marker(o, value)
        }
    }
}

fn is_value_marker(o: &Obj, value: &str) -> bool {
    match o {
        Tree::Node(label, _, _) => label.to_string() == value,
        _ => false,
    }
}

fn contains_value(o: &Obj, value: &str) -> bool {
    match o {
        Tree::Node(label, _, children) => {
            label.to_string() == value || children.iter().any(|child| contains_value(child, value))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(name: &str) -> Arr {
        name.parse().expect("test operation name is valid")
    }

    #[test]
    fn detects_value_marker_recursively() {
        let object = Tree::Node(
            op("*"),
            0,
            vec![
                Tree::Node(op("1"), 0, vec![]),
                Tree::Node(op("val"), 0, vec![Tree::Node(op("f32"), 0, vec![])]),
            ],
        );

        assert!(Erase::with_value("val").is_value(&object));
    }

    #[test]
    fn shallow_detection_ignores_nested_value_marker() {
        let object = Tree::Node(
            op("*"),
            0,
            vec![
                Tree::Node(op("1"), 0, vec![]),
                Tree::Node(op("val"), 0, vec![Tree::Node(op("f32"), 0, vec![])]),
            ],
        );

        assert!(!Erase::with_value_shallow("val").is_value(&object));
    }
}

impl Default for Erase {
    fn default() -> Self {
        Self::default_value()
    }
}

impl Functor<Obj, Arr, Obj, Arr> for Erase {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        // If this op is `bound.eta` [t] → [bound(t), value(t)], replace it with `discard ; cup`
        if source.iter().any(|o| self.is_value(o)) || target.iter().any(|o| self.is_value(o)) {
            OpenHypergraph::singleton(a.clone(), source.to_vec(), target.to_vec())
        } else {
            let mut a = OpenHypergraph::identity(source.to_vec());
            let mut b = OpenHypergraph::identity(target.to_vec());
            a.targets = vec![];
            b.sources = vec![];
            a.tensor(&b)
        }
    }

    fn map_arrow(
        &self,
        f: &open_hypergraphs::lax::OpenHypergraph<Obj, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}
