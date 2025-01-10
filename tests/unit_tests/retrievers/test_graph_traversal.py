import json
from typing import Any

import pytest
from conftest import sorted_doc_ids
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from graph_pancake.retrievers.graph_traversal_retriever import (
    Edge,
    GraphTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.eager import (
    InMemoryTraversalAdapter,
    TraversalAdapter,
)
from tests.embeddings import (
    AnimalEmbeddings,
    EarthEmbeddings,
    ParserEmbeddings,
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
    ) == {"incoming.4": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="place", value="berlin", is_denormalized=True)
    ) == {"place.berlin": True}


def test_traversal() -> None:
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )
    vector_store = InMemoryVectorStore(embedding=EarthEmbeddings())
    vector_store.add_documents([greetings, doc1, doc2])

    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=True,
        ),
        edges=[("outgoing", "incoming"), "keywords"],
        start_k=2,
        depth=2,
    )

    docs = retriever.invoke("Earth", start_k=1, depth=0)
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = retriever.invoke("Earth", depth=0)
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = retriever.invoke("Earth", start_k=1, depth=1)
    assert set(sorted_doc_ids(docs)) == {"doc1", "doc2", "greetings"}


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


class TestGraphTraversal:
    @pytest.fixture(scope="class")
    def animal_docs(self) -> list[Document]:
        documents = []
        with open("tests/data/animals.jsonl", "r") as file:
            for line in file:
                data = json.loads(line.strip())
                documents.append(
                    Document(
                        id=data["id"],
                        page_content=data["text"],
                        metadata=data["metadata"],
                    )
                )

        return documents

    @pytest.fixture(scope="class")
    def animal_vector_store(self, animal_docs: list[Document]) -> VectorStore:
        store = InMemoryVectorStore(embedding=AnimalEmbeddings())
        store.add_documents(animal_docs)
        return store

    def test_invoke_sync(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a vector store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
        vector_store.add_documents(graph_vector_store_docs)

        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(vector_store=vector_store),
            edges=[("out", "in"), "tag"],
            depth=2,
            start_k=2,
        )

        docs = retriever.invoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = retriever.invoke(input="[2, 10]")
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in docs}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

    async def test_invoke_async(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a graph store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
        await vector_store.aadd_documents(graph_vector_store_docs)

        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(vector_store=vector_store),
            edges=[("out", "in"), "tag"],
            depth=2,
            start_k=2,
        )
        docs = await retriever.ainvoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = await retriever.ainvoke(input="[2, 10]")
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

    @pytest.mark.parametrize("support_normalized_metadata", [False, True])
    def test_animals_sync(
        self,
        support_normalized_metadata: bool,
        animal_vector_store: VectorStore,
    ) -> None:
        query = "small agile mammal"

        depth_0_expected = ["fox", "mongoose"]

        # test non-graph search
        docs = animal_vector_store.similarity_search(query, k=2)
        assert sorted_doc_ids(docs) == depth_0_expected

        # test graph-search on a normalized bi-directional edge
        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(
                vector_store=animal_vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=["keywords"],
            start_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert (
            sorted_doc_ids(docs)
            == ["cat", "coyote", "fox", "gazelle", "hyena", "jackal", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )

        docs = retriever.invoke(query, depth=2)
        assert (
            sorted_doc_ids(docs)
            == [
                "alpaca",
                "bison",
                "cat",
                "chicken",
                "cockroach",
                "coyote",
                "crow",
                "dingo",
                "dog",
                "fox",
                "gazelle",
                "horse",
                "hyena",
                "jackal",
                "llama",
                "mongoose",
                "ostrich",
            ]
            if support_normalized_metadata
            else depth_0_expected
        )

        # test graph-search on a standard bi-directional edge
        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(
                vector_store=animal_vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=["habitat"],
            start_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert sorted_doc_ids(docs) == [
            "bobcat",
            "cobra",
            "deer",
            "elk",
            "fox",
            "mongoose",
        ]

        docs = retriever.invoke(query, depth=2)
        assert sorted_doc_ids(docs) == [
            "bobcat",
            "cobra",
            "deer",
            "elk",
            "fox",
            "mongoose",
        ]

        # test graph-search on a standard -> normalized edge
        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(
                vector_store=animal_vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=[("habitat", "keywords")],
            start_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert (
            sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )

        docs = retriever.invoke(query, depth=2)
        assert (
            sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )
