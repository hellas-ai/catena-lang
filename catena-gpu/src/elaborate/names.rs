use hexpr::{Hexpr, LVar, Operation, try_interpret};
use metacat::theory::{
    RawTheorySet, Theory, TheoryId, TheorySet,
    ast::{RawTheory, RawTheoryArrow},
    transitive_dependency_subset,
};
use open_hypergraphs::lax::{NodeId, OpenHypergraph};

use super::ElaborateError;

const NAME_PREFIX: &str = "name.";

#[derive(Default)]
struct Vars(usize);

impl Vars {
    fn next(&mut self, stem: &str) -> Result<LVar, ElaborateError> {
        let name = format!("__catena_{stem}{}", self.0);
        self.0 += 1;
        name.parse()
            .map_err(|_| ElaborateError::InvalidVariable(name))
    }
}

pub fn elaborate_theory(raw: &mut RawTheorySet, name: &Operation) -> Result<(), ElaborateError> {
    let theory = raw
        .theories
        .get(name)
        .ok_or_else(|| ElaborateError::MissingTheory(name.to_string()))?;
    let syntax_name = theory.syntax_category.clone();
    let dependencies = transitive_dependency_subset([syntax_name.clone()], raw)?;
    let interpreted = TheorySet::from_raw(dependencies)?;
    let syntax = interpreted
        .theories
        .get(&TheoryId(syntax_name.clone()))
        .ok_or_else(|| ElaborateError::MissingSyntax(syntax_name.to_string()))?;
    let theory = raw
        .theories
        .get_mut(name)
        .ok_or_else(|| ElaborateError::MissingTheory(name.to_string()))?;
    add_names(theory, syntax)
}

fn add_names(raw: &mut RawTheory, syntax: &Theory) -> Result<(), ElaborateError> {
    let generated = raw
        .arrows
        .values()
        .map(|arrow| name_arrow(syntax, &raw.name, arrow))
        .collect::<Result<Vec<_>, _>>()?;
    for arrow in generated {
        raw.arrows.insert(arrow.name.clone(), arrow);
    }
    Ok(())
}

fn name_arrow(
    syntax: &Theory,
    theory: &Operation,
    arrow: &RawTheoryArrow,
) -> Result<RawTheoryArrow, ElaborateError> {
    let source = try_interpret(&syntax.local_signature(), &arrow.type_maps.0).map_err(|error| {
        ElaborateError::SourceTypeMap {
            theory: theory.to_string(),
            arrow: arrow.name.to_string(),
            map: arrow.type_maps.0.clone(),
            error,
        }
    })?;
    let target = try_interpret(&syntax.local_signature(), &arrow.type_maps.1).map_err(|error| {
        ElaborateError::TargetTypeMap {
            theory: theory.to_string(),
            arrow: arrow.name.to_string(),
            map: arrow.type_maps.1.clone(),
            error,
        }
    })?;
    let mut vars = Vars::default();
    let parameters = (0..source.sources.len())
        .map(|_| vars.next("p"))
        .collect::<Result<Vec<_>, _>>()?;
    let source_map = Hexpr::Frobenius {
        sources: parameters.clone(),
        targets: parameters.clone(),
    };
    let mut copied = parameters.clone();
    copied.extend(parameters.clone());
    let copy = Hexpr::Frobenius {
        sources: parameters,
        targets: copied,
    };
    let packed_source = serialize(&packer(source.targets.len()), &mut vars)?;
    let packed_target = serialize(&packer(target.targets.len()), &mut vars)?;
    let target_map = Hexpr::Composition(vec![
        copy,
        Hexpr::Tensor(vec![
            Hexpr::Composition(vec![arrow.type_maps.0.clone(), packed_source]),
            Hexpr::Composition(vec![arrow.type_maps.1.clone(), packed_target]),
        ]),
        Hexpr::Operation(op("->")?),
        Hexpr::Operation(op("val")?),
    ]);
    Ok(RawTheoryArrow {
        name: op(&format!("{NAME_PREFIX}{}", arrow.name))?,
        type_maps: (source_map, target_map),
        definition: None,
    })
}

fn packer(count: usize) -> OpenHypergraph<(), Operation> {
    let mut graph = OpenHypergraph::empty();
    let sources = (0..count).map(|_| graph.new_node(())).collect::<Vec<_>>();
    graph.sources = sources.clone();
    graph.targets = vec![pack_nodes(&mut graph, &sources)];
    graph
}

fn pack_nodes(graph: &mut OpenHypergraph<(), Operation>, nodes: &[NodeId]) -> NodeId {
    match nodes {
        [] => {
            let output = graph.new_node(());
            graph.new_edge("1".parse().unwrap(), (vec![], vec![output]));
            output
        }
        [only] => *only,
        [head, tail @ ..] => {
            let tail = pack_nodes(graph, tail);
            let output = graph.new_node(());
            graph.new_edge("*".parse().unwrap(), (vec![*head, tail], vec![output]));
            output
        }
    }
}

fn serialize(
    graph: &OpenHypergraph<(), Operation>,
    vars: &mut Vars,
) -> Result<Hexpr, ElaborateError> {
    let node_vars = (0..graph.hypergraph.nodes.len())
        .map(|_| vars.next("w"))
        .collect::<Result<Vec<_>, _>>()?;
    let select = |nodes: &[NodeId]| nodes.iter().map(|node| node_vars[node.0].clone()).collect();
    let mut parts = vec![Hexpr::Frobenius {
        sources: select(&graph.sources),
        targets: vec![],
    }];
    for (operation, edge) in graph
        .hypergraph
        .edges
        .iter()
        .zip(&graph.hypergraph.adjacency)
    {
        parts.push(Hexpr::Composition(vec![
            Hexpr::Frobenius {
                sources: vec![],
                targets: select(&edge.sources),
            },
            Hexpr::Operation(operation.clone()),
            Hexpr::Frobenius {
                sources: select(&edge.targets),
                targets: vec![],
            },
        ]));
    }
    parts.push(Hexpr::Frobenius {
        sources: vec![],
        targets: select(&graph.targets),
    });
    Ok(Hexpr::Composition(parts))
}

fn op(name: &str) -> Result<Operation, ElaborateError> {
    name.parse()
        .map_err(|_| ElaborateError::InvalidOperation(name.to_string()))
}
