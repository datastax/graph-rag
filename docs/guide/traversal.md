# Traversal

At a high level, traversal performs the following steps:

1. Retrieve `start_k` most similar to the `query` using vector search.
2. Find the nodes reachable from the `initial_root_ids`.
3. Discover the `start_k` nodes and the neighbors of the initial roots as "depth 0" candidates.
4. Ask the strategy which nodes to visit next.
5. If no more nodes to visit, exit and return the selected nodes.
6. Record those nodes as selected and retrieve the top `adjacent_k` nodes reachable from them.
7. Discover the newly reachable nodes (updating depths as needed).
8. Goto 4.

## Traversal Methods

The [`graph_retriever`](../reference/graph_retriever/index.md) package provides [`traverse`][graph_retriever.traverse] and [`atraverse`][graph_retriever.atraverse] for performing traversals.

## LangChain Graph Retriever {: #graph-retriever }

The [`langchain_graph_retriever`](../reference/langchain_graph_retriever/index.md) package provides [`GraphRetriever`][langchain_graph_retriever.GraphRetriever], an implementation of LangChain's [`BaseRetriever`][langchain_core.retrievers.BaseRetriever] which performs traversals.