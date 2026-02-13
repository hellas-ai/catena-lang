//! Erase type-level operations

use crate::lang::{Arr, Obj, is_value};
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

#[derive(Clone)]
pub struct Erase;

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
        if source.iter().any(is_value) || target.iter().any(is_value) {
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
