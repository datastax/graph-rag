# Strategies

Strategies determine which nodes are selected during [traversal](./traversal.md).

All strategies allow you to control how many nodes are retrieved (`k`) as well
as how many nodes are found during the initial vector search (`start_k`) and
each step of the traversal (`adjacent_k`) as well as bounding the nodes
retrieved based on depth (`max_depth`).

## Eager

The [`Eager`][graph_retriever.strategies.Eager] strategy selects all of the discovered nodes at each step of the traversal.

It doesn't support configuration beyond the standard options.

## MMR

The [`MMR`][graph_retriever.strategies.Mmr] strategy selects nodes with the
highest maximum marginal relevance score at each iteration.

It can be configured with a `lambda_mult` which controls the trade-off between relevance and diversity.

## Scored

The [`Scored`][graph_retriever.strategies.Scored] strategy applies a user-defined function to each node to assign a score, and selects a number of nodes with the highest scores.

## User-Defined Strategies

You can also implement your own [`Strategy`][graph_retriever.strategies.Strategy]. This allows you to control how discovered nodes are tracked and selected for traversal.