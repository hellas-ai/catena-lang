use metacat::theory::TheorySet;

pub fn summary(theory_set: &TheorySet) -> String {
    let mut lines = vec!["OK: check passed".to_string()];
    for (id, theory) in &theory_set.theories {
        if let metacat::theory::Theory::Theory { arrows, .. } = theory {
            let definitions = arrows
                .values()
                .filter(|arrow| arrow.definition.is_some())
                .count();
            lines.push(format!("  {id}: {definitions} definitions"));
        }
    }
    lines.push(String::new());
    lines.join("\n")
}
