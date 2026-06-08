//! Inline a pre-set list of definitions

use std::collections::HashMap;

use crate::lang::Arr;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

pub struct Inline<O> {
    pub definitions: HashMap<Arr, OpenHypergraph<O, Arr>>,
}

impl<O: Clone> Functor<O, Arr, O, Arr> for Inline<O> {
    fn map_object(&self, o: &O) -> impl ExactSizeIterator<Item = O> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[O],
        target: &[O],
    ) -> open_hypergraphs::lax::OpenHypergraph<O, Arr> {
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
        f: &open_hypergraphs::lax::OpenHypergraph<O, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<O, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}
