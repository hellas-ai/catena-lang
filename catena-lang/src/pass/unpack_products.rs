use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::{
    OpenHypergraph,
    functor::{Functor, try_define_map_arrow},
};

use crate::{pass::PassError, report::TheoryTermMap};

pub type Obj = Tree<(), Operation>;

const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";

const PRODUCT_INTRO: &str = "*.intro";
const PRODUCT_ELIM: &str = "*.elim";
const UNIT_INTRO: &str = "unit.intro";
const UNIT_ELIM: &str = "unit.elim";

#[derive(Clone, Copy, Debug, Default)]
pub struct UnpackProducts;

pub trait UnpackProductOperation {
    fn is_unpack_product_operation(&self) -> bool;
}

impl UnpackProductOperation for Operation {
    fn is_unpack_product_operation(&self) -> bool {
        is_forgotten_operation(self.as_str())
    }
}

impl<A: Clone + UnpackProductOperation> Functor<Obj, A, Obj, A> for UnpackProducts {
    fn map_object(&self, o: &Obj) -> impl ExactSizeIterator<Item = Obj> {
        flatten_object(o).into_iter()
    }

    fn map_operation(&self, a: &A, source: &[Obj], target: &[Obj]) -> OpenHypergraph<Obj, A> {
        let source = map_objects(source);
        let target = map_objects(target);

        if a.is_unpack_product_operation() {
            assert_eq!(
                source, target,
                "unpacked product/unit operation must have matching flattened boundaries"
            );
            return OpenHypergraph::identity(source);
        }

        OpenHypergraph::singleton(a.clone(), source, target)
    }

    fn map_arrow(&self, f: &OpenHypergraph<Obj, A>) -> OpenHypergraph<Obj, A> {
        try_define_map_arrow(self, f).expect("programmer error: unpack-products is not a functor")
    }
}

pub fn run<A: Clone + UnpackProductOperation>(
    terms: &TheoryTermMap<A>,
) -> Result<TheoryTermMap<A>, PassError> {
    terms
        .iter()
        .map(|(theory_id, definitions)| {
            let definitions = definitions
                .iter()
                .map(|(definition_name, term)| {
                    let mut transformed = UnpackProducts.map_arrow(term);
                    transformed.quotient().map_err(|_| PassError::Quotient {
                        pass: "unpack_products",
                        theory: theory_id.to_string(),
                        definition: definition_name.to_string(),
                    })?;
                    Ok((definition_name.clone(), transformed))
                })
                .collect::<Result<_, PassError>>()?;
            Ok((theory_id.clone(), definitions))
        })
        .collect()
}

pub fn flatten_object(o: &Obj) -> Vec<Obj> {
    match o {
        Tree::Empty => vec![],
        Tree::Node(op, _, children) if op.as_str() == UNIT_TYPE && children.is_empty() => vec![],
        Tree::Node(op, _, children) if op.as_str() == PRODUCT_TYPE => {
            children.iter().flat_map(flatten_object).collect()
        }
        _ => vec![o.clone()],
    }
}

pub fn map_objects(objects: &[Obj]) -> Vec<Obj> {
    objects.iter().flat_map(flatten_object).collect()
}

fn is_forgotten_operation(operation: &str) -> bool {
    matches!(
        operation,
        PRODUCT_INTRO | PRODUCT_ELIM | UNIT_INTRO | UNIT_ELIM
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use open_hypergraphs::category::Arrow;

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }

    fn ty(name: &str) -> Obj {
        Tree::Node(op(name), 0, vec![])
    }

    fn product(children: Vec<Obj>) -> Obj {
        Tree::Node(op(PRODUCT_TYPE), 0, children)
    }

    fn unit() -> Obj {
        Tree::Node(op(UNIT_TYPE), 0, vec![])
    }

    #[test]
    fn flattens_products_and_units_on_objects() {
        let object = product(vec![ty("A"), product(vec![unit(), ty("B")])]);

        assert_eq!(flatten_object(&object), vec![ty("A"), ty("B")]);
    }

    #[test]
    fn replaces_product_intro_with_identity_on_flattened_wires() {
        let source = vec![ty("A"), ty("B")];
        let target = vec![product(vec![ty("A"), ty("B")])];

        let mapped = UnpackProducts.map_operation(&op(PRODUCT_INTRO), &source, &target);

        assert!(mapped.hypergraph.edges.is_empty());
        assert_eq!(mapped.source(), vec![ty("A"), ty("B")]);
        assert_eq!(mapped.target(), vec![ty("A"), ty("B")]);
    }

    #[test]
    fn replaces_unit_intro_with_identity_on_empty_boundary() {
        let source = vec![];
        let target = vec![unit()];

        let mapped = UnpackProducts.map_operation(&op(UNIT_INTRO), &source, &target);

        assert!(mapped.hypergraph.edges.is_empty());
        assert_eq!(mapped.source(), vec![]);
        assert_eq!(mapped.target(), vec![]);
    }

    #[test]
    fn preserves_non_structural_operations_with_flattened_boundaries() {
        let source = vec![product(vec![ty("A"), ty("B")])];
        let target = vec![ty("C")];

        let mapped = UnpackProducts.map_operation(&op("f"), &source, &target);

        assert_eq!(mapped.hypergraph.edges, vec![op("f")]);
        assert_eq!(mapped.source(), vec![ty("A"), ty("B")]);
        assert_eq!(mapped.target(), vec![ty("C")]);
    }
}
