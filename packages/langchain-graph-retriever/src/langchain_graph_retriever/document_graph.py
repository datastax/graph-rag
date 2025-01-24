"""Utilities for creating and analyzing a graph of documents."""

from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx
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

    best_modularity = float("-inf")
    best_communities = [[node] for node in graph]
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
    edges: Iterable[str | tuple[str, str]],
) -> nx.DiGraph:
    """
    Create a directed graph from a sequence of documents.

    This function constructs a directed graph where each document is a node, and
    edges are defined based on relationships in the document metadata.

    Parameters
    ----------
    documents : Sequence[Document]
        A sequence of documents to add as nodes.
    edges : Iterable[str | tuple[str, str]])
        Definitions of edges to use for creating the graph. Each edge can be:
        - A string representing a single metadata field (interpreted as both
            `from` and `to`).
        - A tuple of two strings (`from_field`, `to_field`) specifying the
            relationship.

    Returns
    -------
    nx.DiGraph
        The created directed graph with documents as nodes and metadata
        relationships as edges.
    """
    graph = nx.DiGraph()

    # Analyze the edges to determine from and to fields.
    edge_fields = set()
    to_fields = set()

    for edge in edges:
        if isinstance(edge, str):
            edge_fields.add((edge, edge))
            to_fields.add(edge)
        else:
            edge_from, edge_to = edge
            edge_fields.add((edge_from, edge_to))
            to_fields.add(edge_to)

    # First pass -- index documents based on "to_fields" so we can navigate to them.
    documents_by_to_field: dict[tuple[Any, Any], set[str]] = {}
    for document in documents:
        assert document.id is not None
        graph.add_node(document.id, doc=document)

        for to_field in to_fields:
            for to_value in _get_md_values(document.metadata, to_field):
                documents_by_to_field.setdefault((to_field, to_value), set()).add(
                    document.id
                )

    # Second pass -- add edges for each outgoing edge.
    # print(graph.nodes)
    for document in documents:
        for from_field, to_field in edge_fields:
            linked_to = set()
            for from_value in _get_md_values(document.metadata, from_field):
                links = documents_by_to_field.get((to_field, from_value), set())
                linked_to.update(links)

            for target in linked_to:
                if document.id != target:
                    graph.add_edge(document.id, target)

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
