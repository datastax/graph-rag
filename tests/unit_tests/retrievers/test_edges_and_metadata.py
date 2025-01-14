from typing import Any

from langchain_core.documents import Document

from graph_pancake.retrievers.graph_traversal_retriever import (
    Edge,
    GraphTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.eager import (
    TraversalAdapter,
)


class FakeAdapter(TraversalAdapter):
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return []


def test_get_normalized_outgoing_edges() -> None:
    doc = Document(
        page_content="test",
        metadata={
            "place": ["berlin", "paris"],
            "incoming": ["one", 2],
            "outgoing": ["three", 4],
            "boolean": True,
            "string": "pizza",
            "number": 42,
        },
    )

    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=["place", ("outgoing", "incoming"), "boolean", ("number", "string")],
        use_denormalized_metadata=False,
    )

    edges = sorted(retriever._get_outgoing_edges(doc=doc))

    assert edges[0] == Edge(key="boolean", value=True, is_denormalized=False)
    assert edges[1] == Edge(key="incoming", value=4, is_denormalized=False)
    assert edges[2] == Edge(key="incoming", value="three", is_denormalized=False)
    assert edges[3] == Edge(key="place", value="berlin", is_denormalized=False)
    assert edges[4] == Edge(key="place", value="paris", is_denormalized=False)
    assert edges[5] == Edge(key="string", value=42, is_denormalized=False)


def test_get_denormalized_outgoing_edges() -> None:
    doc = Document(
        page_content="test",
        metadata={
            "place.berlin": True,
            "place.paris": True,
            "incoming.one": True,
            "incoming.2": True,
            "outgoing.three": True,
            "outgoing.4": True,
            "boolean": True,
            "string": "pizza",
            "number": 42,
        },
    )

    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=["place", ("outgoing", "incoming"), "boolean", ("number", "string")],
        use_denormalized_metadata=True,
        denormalized_static_value=True,
    )

    edges = sorted(retriever._get_outgoing_edges(doc=doc))

    assert edges[0] == Edge(key="boolean", value=True, is_denormalized=False)
    assert edges[1] == Edge(key="incoming", value="4", is_denormalized=True)
    assert edges[2] == Edge(key="incoming", value="three", is_denormalized=True)
    assert edges[3] == Edge(key="place", value="berlin", is_denormalized=True)
    assert edges[4] == Edge(key="place", value="paris", is_denormalized=True)
    assert edges[5] == Edge(key="string", value=42, is_denormalized=False)


def test_get_normalized_metadata_filter() -> None:
    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=[],
        use_denormalized_metadata=False,
    )

    assert retriever._get_metadata_filter(
        edge=Edge(key="boolean", value=True, is_denormalized=False)
    ) == {"boolean": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="incoming", value=4, is_denormalized=False)
    ) == {"incoming": 4}

    assert retriever._get_metadata_filter(
        edge=Edge(key="place", value="berlin", is_denormalized=False)
    ) == {"place": "berlin"}


def test_get_denormalized_metadata_filter() -> None:
    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=[],
        use_denormalized_metadata=True,
    )

    assert retriever._get_metadata_filter(
        edge=Edge(key="boolean", value=True, is_denormalized=False)
    ) == {"boolean": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="incoming", value=4, is_denormalized=True)
    ) == {"incoming.4": "true"}

    assert retriever._get_metadata_filter(
        edge=Edge(key="place", value="berlin", is_denormalized=True)
    ) == {"place.berlin": "true"}
