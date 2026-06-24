use open_hypergraphs::lax::OpenHypergraph;
use open_hypergraphs::lax::functor::Functor;

use crate::{
    pass::{
        forget_closures::ForgetClosures,
        forget_intro_elim_units::ForgetIntroElimUnits,
        record_object_sizes::{OperationWithSizes, RecordObjectSizes},
    },
    report::AnnotatedTerm,
};

pub fn apply(term: &AnnotatedTerm) -> AnnotatedTerm<OperationWithSizes<hexpr::Operation>> {
    let term = quotient(ForgetClosures.map_arrow(term));
    let term = quotient(RecordObjectSizes.map_arrow(&term));
    ForgetIntroElimUnits.map_arrow(&term)
}

fn quotient<A: Clone>(
    mut term: OpenHypergraph<metacat::tree::Tree<(), hexpr::Operation>, A>,
) -> OpenHypergraph<metacat::tree::Tree<(), hexpr::Operation>, A> {
    term.quotient()
        .expect("pass output should quotient successfully");
    term
}
