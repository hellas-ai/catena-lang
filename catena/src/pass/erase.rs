//! Erase type-level operations

use crate::lang::{Arr, Obj, is_value};
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

#[derive(Clone)]
pub struct Erase {
    value: Option<&'static str>,
}

impl Erase {
    pub fn default_value() -> Self {
        Self { value: None }
    }

    pub fn with_value(value: &'static str) -> Self {
        Self { value: Some(value) }
    }

    fn is_value(&self, o: &Obj) -> bool {
        match self.value {
            Some(value) => match o {
                Tree::Node(label, _, _) => label.to_string() == value,
                _ => false,
            },
            None => is_value(o),
        }
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
