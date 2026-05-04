//! Forget the `bound` symbol, replacing it with `value`.
use crate::lang::{Arr, Obj};
use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

/// Quotient forward and reverse values in the language:
/// any type labeled "bound(t)" becomes "value(t)".
#[derive(Clone)]
pub struct ForgetBound {
    bound_key: Operation,
    value_key: Operation,
}

impl ForgetBound {
    pub fn new(bound_key: Operation, value_key: Operation) -> Self {
        Self {
            bound_key,
            value_key,
        }
    }
}

impl Functor<Obj, Arr, Obj, Arr> for ForgetBound {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        // bound(t) ⇒ value(t)
        let o: Tree<_, Operation> = match o {
            Tree::Node(label, port, trees) => {
                let new_label = if label == &self.bound_key {
                    self.value_key.clone()
                } else {
                    label.clone()
                };
                Tree::Node(new_label, *port, trees.clone())
            }
            x => x.clone(),
        };

        std::iter::once(o)
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        let source = source.iter().flat_map(|o| self.map_object(o)).collect();
        let target = target.iter().flat_map(|o| self.map_object(o)).collect();
        OpenHypergraph::singleton(a.clone(), source, target)
    }

    fn map_arrow(
        &self,
        f: &open_hypergraphs::lax::OpenHypergraph<Obj, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}
