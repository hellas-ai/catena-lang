//! Inline a pre-set list of definitions

use std::collections::HashMap;

use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

pub struct Inline {
    pub definitions: HashMap<Arr, OpenHypergraph<Obj, Arr>>,
}

impl Functor<Obj, Arr, Obj, Arr> for Inline {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        match self.definitions.get(a) {
            Some(f) => f.clone(),
            None => {
                let source = source.iter().flat_map(|o| self.map_object(o)).collect();
                let target = target.iter().flat_map(|o| self.map_object(o)).collect();
                OpenHypergraph::singleton(a.clone(), source, target)
            }
        }
    }

    fn map_arrow(
        &self,
        f: &open_hypergraphs::lax::OpenHypergraph<Obj, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}
