from typing import Any

import pytest
from langchain_graph_retriever.edges.metadata import MetadataEdgeFunction
from langchain_graph_retriever.types import Edges, MetadataEdge, Node


def mk_node(metadata: dict[str, Any]) -> Node:
    return Node(
        id="id",
        metadata=metadata,
        depth=0,
        embedding=[],
    )


def test_initialization():
    edge_function = MetadataEdgeFunction(["a", ("b", "c"), "b"])
    assert edge_function.edges == [("a", "a"), ("b", "c"), ("b", "b")]


def test_normalized():
    edge_function = MetadataEdgeFunction(
        [("href", "url")], use_normalized_metadata=True
    )
    assert edge_function(mk_node({"href": "a", "url": "b"})) == Edges(
        {MetadataEdge("url", "b")},
        {MetadataEdge("url", "a")},
    )

    assert edge_function(mk_node({"href": ["a", "c"], "url": "b"})) == Edges(
        {MetadataEdge("url", "b")},
        {MetadataEdge("url", "a"), MetadataEdge("url", "c")},
    )

    assert edge_function(mk_node({"href": ["a", "c"], "url": ["b", "d"]})) == Edges(
        {MetadataEdge("url", "b"), MetadataEdge("url", "d")},
        {MetadataEdge("url", "a"), MetadataEdge("url", "c")},
    )


def test_denormalized():
    edge_function = MetadataEdgeFunction(
        [("href", "url")],
        use_normalized_metadata=False,
        # Use non-default values so we can verify the fields are used.
        denormalized_path_delimiter=":",
        denormalized_static_value=57,
    )
    assert edge_function(mk_node({"href": "a", "url": "b"})) == Edges(
        {MetadataEdge("url", "b")},
        {MetadataEdge("url", "a")},
    )

    assert edge_function(mk_node({"href:a": 57, "href:c": 57, "url": "b"})) == Edges(
        {MetadataEdge("url", "b")}, {MetadataEdge("url", "a"), MetadataEdge("url", "c")}
    )

    assert edge_function(
        mk_node(
            {
                "href:a": 57,
                "href:c": 57,
                "url:b": 57,
                "url:d": 57,
            }
        )
    ) == Edges(
        {MetadataEdge("url", "b"), MetadataEdge("url", "d")},
        {MetadataEdge("url", "a"), MetadataEdge("url", "c")},
    )


def test_unsupported_values():
    edge_function = MetadataEdgeFunction(
        [("href", "url")],
        use_normalized_metadata=True,
    )

    # Unsupported value
    with pytest.warns(UserWarning, match=r"Unsupported value .* in 'href'"):
        edge_function(mk_node({"href": None}))

    # Unsupported item value
    with pytest.warns(UserWarning, match=r"Unsupported item value .* in 'href'"):
        edge_function(mk_node({"href": [None]}))

    edge_function = MetadataEdgeFunction(
        [("href", "url")],
        use_normalized_metadata=False,
        # Use non-default values so we can verify the fields are used.
        denormalized_path_delimiter=":",
        denormalized_static_value=57,
    )
    # Unsupported value
    with pytest.warns(UserWarning, match=r"Unsupported value .* in 'href'"):
        edge_function(mk_node({"href": None}))

    # It is OK for the list to exist in the metadata, although we do issue a warning
    # for that case.
    with pytest.warns(UserWarning, match="Normalized value [[]'a', 'c'] in 'href'"):
        assert edge_function(
            mk_node(
                {
                    "href": ["a", "c"],
                }
            )
        ) == Edges(set(), set())
