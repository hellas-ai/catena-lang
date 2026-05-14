//! Forget pseudo-close loopback structure after typechecking.
//!
//! `cap`/`cup` are useful typechecking markers for an explicit feedback wire,
//! but the structured compiler should see only the underlying wiring.

use crate::lang::{Arr, Obj};
use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::{
    lax::OpenHypergraph,
    lax::functor::{Functor, try_define_map_arrow},
    strict::vec::FiniteFunction,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetLoopbackConfig {
    pub cap: &'static str,
    pub cup: &'static str,
    pub loopback: &'static str,
    pub value: &'static str,
}

impl Default for ForgetLoopbackConfig {
    fn default() -> Self {
        Self {
            cap: "cap",
            cup: "cup",
            loopback: "loopback",
            value: "val",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForgetLoopback {
    config: ForgetLoopbackConfig,
}

impl ForgetLoopback {
    pub fn new(config: ForgetLoopbackConfig) -> Self {
        Self { config }
    }

    pub fn default_control() -> Self {
        Self::new(ForgetLoopbackConfig::default())
    }

    pub fn config(&self) -> &ForgetLoopbackConfig {
        &self.config
    }

    fn is_cap(&self, a: &Arr) -> bool {
        a.to_string() == self.config.cap
    }

    fn is_cup(&self, a: &Arr) -> bool {
        a.to_string() == self.config.cup
    }

    fn normalize_loopback_wire(&self, o: &Obj) -> Obj {
        match o {
            Tree::Node(label, _, children)
                if label.to_string() == self.config.loopback && children.len() == 1 =>
            {
                self.normalize_loopback_wire(&children[0])
            }
            Tree::Node(label, _, _) if label.to_string() == self.config.value => o.clone(),
            Tree::Empty => Tree::Empty,
            _ => Tree::Node(value_operation(self.config.value), 0, vec![o.clone()]),
        }
    }

    fn is_value_marker(&self, o: &Obj) -> bool {
        match o {
            Tree::Node(label, _, _) => label.to_string() == self.config.value,
            _ => false,
        }
    }

    fn loopback_wire_type(&self, source: &[Obj], target: &[Obj]) -> Option<Obj> {
        let labels = source
            .iter()
            .chain(target)
            .map(|o| self.normalize_loopback_wire(o))
            .collect::<Vec<_>>();
        labels
            .iter()
            .find(|o| self.is_value_marker(o))
            .cloned()
            .or_else(|| labels.first().cloned())
    }
}

impl Functor<Obj, Arr, Obj, Arr> for ForgetLoopback {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        std::iter::once(self.normalize_loopback_wire(o))
    }

    fn map_operation(
        &self,
        a: &Arr,
        source: &[Obj],
        target: &[Obj],
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        if self.is_cap(a) || self.is_cup(a) {
            return loopback_spider(
                source.len(),
                target.len(),
                self.loopback_wire_type(source, target),
            );
        }

        OpenHypergraph::singleton(
            a.clone(),
            source
                .iter()
                .map(|o| self.normalize_loopback_wire(o))
                .collect(),
            target
                .iter()
                .map(|o| self.normalize_loopback_wire(o))
                .collect(),
        )
    }

    fn map_arrow(
        &self,
        f: &open_hypergraphs::lax::OpenHypergraph<Obj, Arr>,
    ) -> open_hypergraphs::lax::OpenHypergraph<Obj, Arr> {
        try_define_map_arrow(self, f).expect("programmer error: not a functor")
    }
}

fn loopback_spider(
    source_count: usize,
    target_count: usize,
    wire_type: Option<Obj>,
) -> OpenHypergraph<Obj, Arr> {
    let Some(wire_type) = wire_type else {
        return OpenHypergraph::empty();
    };

    OpenHypergraph::spider(
        FiniteFunction::terminal(source_count),
        FiniteFunction::terminal(target_count),
        vec![wire_type],
    )
    .expect("terminal spiders are well-formed")
}

fn value_operation(value: &str) -> Operation {
    value
        .parse()
        .expect("programmer error: invalid configured value operation")
}
