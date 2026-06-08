//! Map `bound.eta` operations into compact-closed structure

use crate::lang::{Arr, Obj};
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

#[derive(Clone)]
pub struct ExpandEta;

impl Functor<Obj, Arr, Obj, Arr> for ExpandEta {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(o.clone())
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        // Map sources/targets
        let source = source.iter().flat_map(|o| self.map_object(o)).collect();

        // NOTE: take *every other* target - inputs are paired
        // (bound(t)●value(t)●bound(u)●value(u)...)
        let target = target
            .iter()
            .step_by(2)
            .flat_map(|o| self.map_object(o))
            .collect();

        // bound.eta ⇒ discard ; cup
        if a.to_string() == "bound.eta" {
            let mut a = OpenHypergraph::identity(source);
            let mut b = OpenHypergraph::identity(target);
            a.targets = vec![];
            b.sources = vec![];
            let mut result = a.tensor(&b);
            result.targets = [result.targets.clone(), result.targets].concat();
            result
        } else {
            OpenHypergraph::singleton(a.clone(), source, target)
        }
    }

    fn map_arrow(
        &self,
        f: &open_hypergraphs::lax::OpenHypergraph<Obj, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}
