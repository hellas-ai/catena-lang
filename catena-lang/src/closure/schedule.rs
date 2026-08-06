//! Explicit ordering dependencies between closure regions in one graph snapshot.
//!
//! A closure marker delimits a region containing the operations which compute
//! a closure result from its argument. Regions can depend on one another for
//! two different reasons.
//!
//! ## `NestedRegion`
//!
//! A `NestedRegion` dependency means that one closure region is structurally
//! inside another. In this deliberately simplified graph, the nested marker's
//! domain, body, and result occur while computing the enclosing body:
//!
//! ```text
//! enclosing-argument
//!         │
//!         v
//!   [ enclosing prefix ]
//!         │
//!         ├────> [ nested body ] ────┐
//!         │                          │
//!         └────────> !closure <──────┘──> nested-closure
//!                                            │
//!                                            v
//!                                    [ enclosing suffix ]
//!                                            │
//!                                            v
//!                                      enclosing-result
//! ```
//!
//! The enclosing region cannot be extracted while the nested marker still
//! delimits part of its computation. The dependency therefore has the order:
//!
//! ```text
//! enclosing region ──NestedRegion──> nested region
//! ```
//!
//! For example, in high-level pseudocode:
//!
//! ```text
//! buildProcessor(scale) =
//!     closure process(value) {               // enclosing region
//!         clampToZero =
//!             closure (x) {                  // nested region
//!                 max(x, 0)
//!             }
//!
//!         clampToZero(value * scale)
//!     }
//! ```
//!
//! `clampToZero` is constructed as part of the computation of `process`, so its
//! closure marker is structurally inside the `process` region. Conversion must
//! first replace `clampToZero` with its closure representation. After region
//! discovery runs again, `process` can be extracted without a nested marker in
//! its body.
//!
//! The same dependency is recorded when an ordinary edge belonging to one
//! region touches an internal wire of another region. That incidence is a
//! structural relationship between the regions. If no region owns the external
//! edge, the internal wire has genuinely escaped and region construction is
//! invalid.
//!
//! The nested region is converted first. Region discovery then runs again on
//! the rewritten graph before the enclosing region is considered ready.
//!
//! ## `CapturedClosure`
//!
//! A `CapturedClosure` dependency does not require structural nesting. First,
//! a producer region creates an opaque closure value:
//!
//! ```text
//! producer-argument ──> [ producer body ] ──> producer-result
//!          │                                      │
//!          └──────────> !closure <────────────────┘──> captured-closure
//! ```
//!
//! A separate capturing region receives that value as part of its environment:
//!
//! ```text
//! captured-closure ──────────────────────────┐
//! capturing-argument ─> [ capturing body ] ──┴─> capturing-result
//!          │                                          │
//!          └──────────> !closure <────────────────────┘──> closure
//! ```
//!
//! These can be sibling regions in the hypergraph. Their dependency is caused
//! by the value flowing between them:
//!
//! ```text
//! capturing region ──CapturedClosure──> producer region
//! ```
//!
//! For example, in high-level pseudocode:
//!
//! ```text
//! buildChooser(useNegative) =
//!     positive =
//!         closure (value) {                   // producer region
//!             abs(value)
//!         }
//!
//!     negative =
//!         closure (value) {                   // producer region
//!             -abs(value)
//!         }
//!
//!     closure choose(value) {                 // capturing region
//!         if useNegative {
//!             negative(value)
//!         } else {
//!             positive(value)
//!         }
//!     }
//! ```
//!
//! `choose` captures the closure values `positive` and `negative`. Their bodies
//! are not structurally inside the `choose` region; their opaque closure values
//! enter `choose` through its environment. Both producer regions must be
//! converted before `choose`, so the environment of `choose` is built from the
//! final closure representations rather than the old opaque values.
//!
//! Closure conversion expands the captured value into its runtime
//! representation:
//!
//! ```text
//! captured-closure
//!     ==>
//! (captured-environment, captured-function-pointer)
//! ```
//!
//! The producer region must be converted first so rediscovery sees both runtime
//! components as inputs of the capturing region. Converting the capturing
//! region first would make its context projection copy the old single opaque
//! value, which would no longer match the expanded pair.
//!
//! The dependency graph must be acyclic. Recursive closure construction is not
//! supported, and strict structural containment cannot be cyclic. A cycle
//! therefore indicates a bug in region discovery or dependency construction.
//!
use std::collections::BTreeSet;

use open_hypergraphs::lax::{EdgeId, NodeId};

use super::region::ClosureRegion;
use crate::pass::forget_closures::{ClosureForgotten, ClosureForgottenTerm};

/// Snapshot-local identity of a region.
///
/// Rewriting and quotienting renumber graph identifiers, so this identity must
/// never be retained across conversion steps. The scheduler rediscovers all
/// regions and rebuilds their dependencies after every replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RegionId {
    pub(super) marker: EdgeId,
}

impl From<&ClosureRegion> for RegionId {
    fn from(region: &ClosureRegion) -> Self {
        Self {
            marker: region.marker,
        }
    }
}

impl PartialOrd for RegionId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RegionId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.marker.0.cmp(&other.marker.0)
    }
}

/// Why one region must be converted before another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RegionDependencyKind {
    /// The prerequisite marker or body is structurally connected inside the
    /// dependent region. It must be removed before the dependent is cut out.
    NestedRegion,

    /// The dependent captures the opaque closure value produced by the
    /// prerequisite. The producer must first expand that value into its
    /// `(Environment, FnPointer)` representation.
    CapturedClosure { closure: NodeId },
}

/// A directed ordering constraint: `dependent` waits for `prerequisite`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RegionDependency {
    pub(super) dependent: RegionId,
    pub(super) prerequisite: RegionId,
    pub(super) kind: RegionDependencyKind,
}

/// Complete dependency analysis for the regions of one definition.
#[derive(Debug)]
pub(super) struct RegionDependencies {
    dependencies: Vec<RegionDependency>,
}

impl RegionDependencies {
    /// A region is ready exactly when it has no unresolved prerequisites.
    pub(super) fn is_ready(&self, region: &ClosureRegion) -> bool {
        let id = RegionId::from(region);
        !self
            .dependencies
            .iter()
            .any(|dependency| dependency.dependent == id)
    }

    /// Return the first ready region in marker order.
    ///
    /// A non-empty region set without a ready member contains a dependency
    /// cycle. Recursive closure construction is unsupported, so such a cycle
    /// is a compiler invariant failure rather than a user-facing error.
    pub(super) fn require_ready_region<'a>(
        &self,
        regions: &'a [ClosureRegion],
    ) -> &'a ClosureRegion {
        regions
            .iter()
            .find(|region| self.is_ready(region))
            .unwrap_or_else(|| {
                panic!(
                    "closure region dependencies must be acyclic: {:#?}",
                    self.dependencies
                )
            })
    }
}

/// Build all known ordering constraints for one current graph snapshot.
///
/// An internal wire used by an edge owned by another region creates a
/// `NestedRegion` dependency. If no region owns that edge, region construction
/// is inconsistent and this function panics with the broken incidence.
pub(super) fn analyze(
    definition: &ClosureForgottenTerm,
    regions: &[ClosureRegion],
) -> RegionDependencies {
    let mut dependencies = Vec::new();

    for dependent in regions {
        add_captured_closure_dependencies(&mut dependencies, dependent, regions);
        add_nested_region_dependencies(&mut dependencies, definition, dependent, regions);
    }

    RegionDependencies { dependencies }
}

fn add_captured_closure_dependencies(
    dependencies: &mut Vec<RegionDependency>,
    dependent: &ClosureRegion,
    regions: &[ClosureRegion],
) {
    for prerequisite in regions {
        if prerequisite.marker == dependent.marker
            || !dependent.environment.contains(&prerequisite.closure)
        {
            continue;
        }
        push_dependency(
            dependencies,
            RegionDependency {
                dependent: dependent.into(),
                prerequisite: prerequisite.into(),
                kind: RegionDependencyKind::CapturedClosure {
                    closure: prerequisite.closure,
                },
            },
        );
    }
}

fn add_nested_region_dependencies(
    dependencies: &mut Vec<RegionDependency>,
    definition: &ClosureForgottenTerm,
    dependent: &ClosureRegion,
    regions: &[ClosureRegion],
) {
    let environment = dependent
        .environment
        .iter()
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();
    let internal = dependent
        .nodes
        .iter()
        .copied()
        .filter(|node| !environment.contains(&node.0) && *node != dependent.closure)
        .map(|node| node.0)
        .collect::<BTreeSet<_>>();

    for prerequisite in regions {
        if prerequisite.marker == dependent.marker {
            continue;
        }
        if internal.contains(&prerequisite.domain.0) || internal.contains(&prerequisite.codomain.0)
        {
            push_dependency(
                dependencies,
                RegionDependency {
                    dependent: dependent.into(),
                    prerequisite: prerequisite.into(),
                    kind: RegionDependencyKind::NestedRegion,
                },
            );
        }
    }

    let mut included = vec![false; definition.hypergraph.edges.len()];
    for edge in dependent.edges.iter().chain([&dependent.marker]) {
        included[edge.0] = true;
    }

    for (index, boundary) in definition.hypergraph.adjacency.iter().enumerate() {
        if included[index]
            || matches!(
                definition.hypergraph.edges[index],
                ClosureForgotten::ClosureMarker
            )
        {
            continue;
        }

        for &node in boundary.sources.iter().chain(&boundary.targets) {
            if !internal.contains(&node.0) {
                continue;
            }
            let edge = EdgeId(index);
            let owners = regions
                .iter()
                .filter(|region| region.marker != dependent.marker && region.edges.contains(&edge))
                .collect::<Vec<_>>();
            assert!(
                !owners.is_empty(),
                "internal wire w{} of closure region e{} escapes through unowned edge e{} (`{}`)",
                node.0,
                dependent.marker.0,
                index,
                definition.hypergraph.edges[index],
            );
            for prerequisite in owners {
                push_dependency(
                    dependencies,
                    RegionDependency {
                        dependent: dependent.into(),
                        prerequisite: prerequisite.into(),
                        kind: RegionDependencyKind::NestedRegion,
                    },
                );
            }
        }
    }
}

fn push_dependency(dependencies: &mut Vec<RegionDependency>, dependency: RegionDependency) {
    if !dependencies.contains(&dependency) {
        dependencies.push(dependency);
    }
}

#[cfg(test)]
mod tests {
    use metacat::tree::Tree;

    use super::*;

    #[test]
    fn captured_closure_makes_the_consumer_wait_for_its_producer() {
        let mut term = ClosureForgottenTerm::empty();
        let producer = empty_region(&mut term);
        let mut consumer = empty_region(&mut term);
        consumer.environment.push(producer.closure);
        consumer.nodes.push(producer.closure);

        let dependencies = analyze(&term, &[producer.clone(), consumer.clone()]);

        assert!(dependencies.is_ready(&producer));
        assert!(!dependencies.is_ready(&consumer));
        assert!(dependencies.dependencies.iter().any(|dependency| {
            dependency.dependent == RegionId::from(&consumer)
                && dependency.prerequisite == RegionId::from(&producer)
                && dependency.kind
                    == RegionDependencyKind::CapturedClosure {
                        closure: producer.closure,
                    }
        }));
    }

    #[test]
    #[should_panic(expected = "closure region dependencies must be acyclic")]
    fn cyclic_closure_captures_are_an_invariant_failure() {
        let mut term = ClosureForgottenTerm::empty();
        let mut first = empty_region(&mut term);
        let mut second = empty_region(&mut term);
        first.environment.push(second.closure);
        first.nodes.push(second.closure);
        second.environment.push(first.closure);
        second.nodes.push(first.closure);
        let regions = [first, second];

        let dependencies = analyze(&term, &regions);
        dependencies.require_ready_region(&regions);
    }

    #[test]
    fn region_owned_internal_use_is_a_nested_region_dependency() {
        let mut term = ClosureForgottenTerm::empty();
        let domain = term.new_node(Tree::Empty);
        let internal = term.new_node(Tree::Empty);
        let codomain = term.new_node(Tree::Empty);
        let dependent_closure = term.new_node(Tree::Empty);
        let nested_result = term.new_node(Tree::Empty);
        let nested_closure = term.new_node(Tree::Empty);
        let first = term.new_edge(
            ClosureForgotten::Operation("first".parse().unwrap()),
            (vec![domain], vec![internal]),
        );
        let second = term.new_edge(
            ClosureForgotten::Operation("second".parse().unwrap()),
            (vec![internal], vec![codomain]),
        );
        let nested_edge = term.new_edge(
            ClosureForgotten::Operation("nested".parse().unwrap()),
            (vec![internal], vec![nested_result]),
        );
        let dependent_marker = term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![dependent_closure]),
        );
        let nested_marker = term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![internal, nested_result], vec![nested_closure]),
        );
        let dependent = ClosureRegion {
            marker: dependent_marker,
            domain,
            codomain,
            closure: dependent_closure,
            environment: vec![],
            nodes: vec![domain, internal, codomain],
            edges: vec![first, second],
        };
        let nested = ClosureRegion {
            marker: nested_marker,
            domain: internal,
            codomain: nested_result,
            closure: nested_closure,
            environment: vec![internal],
            nodes: vec![internal, nested_result],
            edges: vec![nested_edge],
        };

        let dependencies = analyze(&term, &[dependent.clone(), nested.clone()]);

        assert!(dependencies.dependencies.iter().any(|dependency| {
            dependency.dependent == RegionId::from(&dependent)
                && dependency.prerequisite == RegionId::from(&nested)
                && dependency.kind == RegionDependencyKind::NestedRegion
        }));
    }

    #[test]
    #[should_panic(expected = "escapes through unowned edge")]
    fn external_internal_use_is_a_construction_invariant_failure() {
        let mut term = ClosureForgottenTerm::empty();
        let domain = term.new_node(Tree::Empty);
        let internal = term.new_node(Tree::Empty);
        let codomain = term.new_node(Tree::Empty);
        let closure = term.new_node(Tree::Empty);
        let escaped = term.new_node(Tree::Empty);
        let first = term.new_edge(
            ClosureForgotten::Operation("first".parse().unwrap()),
            (vec![domain], vec![internal]),
        );
        let second = term.new_edge(
            ClosureForgotten::Operation("second".parse().unwrap()),
            (vec![internal], vec![codomain]),
        );
        term.new_edge(
            ClosureForgotten::Operation("outside".parse().unwrap()),
            (vec![internal], vec![escaped]),
        );
        let marker = term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![closure]),
        );
        let region = ClosureRegion {
            marker,
            domain,
            codomain,
            closure,
            environment: vec![],
            nodes: vec![domain, internal, codomain],
            edges: vec![first, second],
        };

        analyze(&term, &[region]);
    }

    fn empty_region(term: &mut ClosureForgottenTerm) -> ClosureRegion {
        let domain = term.new_node(Tree::Empty);
        let codomain = term.new_node(Tree::Empty);
        let closure = term.new_node(Tree::Empty);
        let marker = term.new_edge(
            ClosureForgotten::ClosureMarker,
            (vec![domain, codomain], vec![closure]),
        );
        ClosureRegion {
            marker,
            domain,
            codomain,
            closure,
            environment: vec![],
            nodes: vec![domain, codomain],
            edges: vec![],
        }
    }
}
