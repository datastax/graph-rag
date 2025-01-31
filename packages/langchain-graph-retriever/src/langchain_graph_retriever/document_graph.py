"""Utilities for creating and analyzing a graph of documents."""

from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx
from graph_retriever import Edge, EdgeFunction
from graph_retriever.content import Content
from graph_retriever.edges.metadata import EdgeSpec, MetadataEdgeFunction
from langchain_core.documents import Document


def _best_communities(graph: nx.DiGraph) -> list[list[str]]:
    """
    Compute the best communities in a directed graph.

    This function iteratively applies the Girvan-Newman algorithm to partition the
    graph into communities. It continues until the modularity no longer improves.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph to analyze.

    Returns
    -------
    list[list[str]]
        A list of communities, where each community is a list of node IDs.
    """
    # TODO: Also continue running until the size of communities is below
    # a specified threshold?

    best_communities = [[node] for node in graph]
    if graph.number_of_edges() == 0:
        # If there are no edges, then we can't do any beter.
        return best_communities

    # TODO: stop running if we reach a target modularity and/or number of communities?
    best_modularity = float("-inf")
    for new_communities in nx.algorithms.community.girvan_newman(graph):
        new_modularity = nx.algorithms.community.modularity(graph, new_communities)
        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_communities = new_communities
        else:
            break
    return best_communities


def _get_md_values(metadata: dict[str, Any], field: str) -> Iterable[Any]:
    """
    Retrieve metadata values for a specific field.

    This function extracts values from the metadata dictionary for the given field,
    handling cases where the value is a single string, a list, or another iterable.

    Parameters
    ----------
    metadata : dict[str, Any]
        The metadata dictionary.
    field : str
        The field to extract values from.

    Returns
    -------
    Iterable[Any]
        A list of values for the specified field. If no values are found, an
        empty list is returned.
    """
    value = metadata.get(field, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if field not in metadata:
        return []
    try:
        return list(value)
    except TypeError as _:
        return [value]


def create_graph(
    documents: Sequence[Document],
    *,
    edges: list[EdgeSpec] | EdgeFunction,
) -> nx.DiGraph:
    """
    Create a directed graph from a sequence of documents.

    This function constructs a directed graph where each document is a node, and
    edges are defined based on relationships in the document metadata.

    Parameters
    ----------
    documents : Sequence[Document]
        A sequence of documents to add as nodes.
    edges : list[EdgeSpec] | EdgeFunction
        Definitions of edges to use for creating the graph or edge function to use.

    Returns
    -------
    nx.DiGraph
        The created directed graph with documents as nodes and metadata
        relationships as edges.

    Raises
    ------
    ValueError
        If the edges are invalid.
    """
    graph = nx.DiGraph()

    edge_function: EdgeFunction
    if isinstance(edges, list):
        edge_function = MetadataEdgeFunction(edges)
    elif callable(edges):
        edge_function = edges
    else:
        raise ValueError(f"Expected `list[EdgeSpec] | EdgeFunction` but got: {edges}")

    # First pass -- index documents based on "to_fields" so we can navigate to them.
    documents_by_incoming: dict[Edge, set[str]] = {}
    outgoing_by_id: dict[str, set[Edge]] = {}
    for document in documents:
        assert document.id is not None
        graph.add_node(document.id, doc=document)

        document_edges = edge_function(
            Content(
                id=document.id,
                content=document.page_content,
                embedding=[],
                metadata=document.metadata,
            )
        )

        for incoming in document_edges.incoming:
            documents_by_incoming.setdefault(incoming, set()).add(document.id)
        outgoing_by_id[document.id] = document_edges.outgoing

    # Second pass -- add edges for each outgoing edge.
    # print(graph.nodes)
    for id, outgoing in outgoing_by_id.items():
        linked_to = set()
        for out in outgoing:
            linked_to.update(documents_by_incoming.get(out, set()))
        for target in linked_to - {id}:
            graph.add_edge(id, target)

    return graph


def group_by_community(graph: nx.DiGraph) -> list[list[Document]]:
    """
    Group documents by community inferred from the graph's structure.

    This function partitions the graph into communities using the Girvan-Newman
    algorithm and groups documents based on their community memberships.

    Paramaters
    ----------
    graph : nx.DiGraph
        The directed graph of documents.

    Returns
    -------
    list[list[Document]]
        A list of communities, where each community is a list of documents.
    """
    # Find communities and output documents grouped by community.
    communities = _best_communities(graph)
    return [[graph.nodes[n]["doc"] for n in community] for community in communities]
