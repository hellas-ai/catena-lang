//! Map `bound.eta` operations into compact-closed structure

use crate::lang::{Arr, Obj};
use metacat::{theory::OperationKey, tree::Tree};
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

#[derive(Clone)]
pub struct Unbound;

impl Functor<Obj, Arr, Obj, Arr> for Unbound {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        // bound(t) ⇒ value(t)
        let o: Tree<_, OperationKey> = match o {
            Tree::Node(label, port, trees) => {
                let new_label = if label.to_string() == "bound" {
                    let result = OperationKey("value".to_string().parse().unwrap());
                    result
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
