use open_hypergraphs::lax::functor::Functor;

use crate::{
    pass::{
        forget_closures::ForgetClosures, forget_intro_elim_units::ForgetIntroElimUnits,
        record_object_sizes::RecordObjectSizes,
    },
    report::{AnnotatedTerm, SizedAnnotatedTerm},
};

pub fn apply(term: &AnnotatedTerm) -> SizedAnnotatedTerm {
    let term = ForgetClosures.map_arrow(term);
    let term = RecordObjectSizes.map_arrow(&term);
    ForgetIntroElimUnits.map_arrow(&term)
}
