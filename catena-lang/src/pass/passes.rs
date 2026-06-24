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
    let term = ForgetClosures.map_arrow(term);
    let term = RecordObjectSizes.map_arrow(&term);
    ForgetIntroElimUnits.map_arrow(&term)
}
