"""Test of Graph Traversal Retriever"""

import pytest
from langchain_core.documents import Document

from graph_pancake.retrievers.graph_traversal_retriever import GraphTraversalRetriever
from tests.embeddings.simple_embeddings import (
    EarthEmbeddings,
)
from tests.integration_tests.retrievers.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.integration_tests.retrievers.conftest import (
    assert_document_format,
    sorted_doc_ids,
)
from tests.integration_tests.stores import StoreFactory, Stores


@pytest.fixture(scope="module")
def earth_store(request: pytest.FixtureRequest, store_factory: StoreFactory) -> Stores:
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

    return store_factory.create(
        request, EarthEmbeddings(), docs=[greetings, doc1, doc2]
    )


async def test_traversal(
    earth_store: Stores,
    support_normalized_metadata: bool,
    invoker,
) -> None:
    retriever = GraphTraversalRetriever(
        store=earth_store.eager,
        edges=[("outgoing", "incoming"), "keywords"],
        start_k=2,
        depth=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs: list[Document] = await invoker(retriever, "Earth", start_k=1, depth=0)
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = await invoker(retriever, "Earth", depth=0)
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = await invoker(retriever, "Earth", start_k=1, depth=1)
    assert sorted_doc_ids(docs) == ["doc1", "doc2", "greetings"]


async def test_invoke(
    parser_store: Stores,
    invoker,
) -> None:
    """Graph traversal search on a vector store."""
    retriever = GraphTraversalRetriever(
        store=parser_store.eager,
        edges=[("out", "in"), "tag"],
        depth=2,
        start_k=2,
    )

    docs: list[Document] = await invoker(retriever, input="[2, 10]", depth=0)
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await invoker(retriever, input="[2, 10]")
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


async def test_animals(
    animal_store: Stores,
    invoker,
    support_normalized_metadata: bool,
) -> None:
    # test graph-search on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=animal_store.eager,
        edges=["keywords"],
        start_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs: list[Document] = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
    assert sorted_doc_ids(docs) == [
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

    # test graph-search on a standard bi-directional edge
    retriever = GraphTraversalRetriever(
        store=animal_store.eager,
        edges=["habitat"],
        start_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
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
        store=animal_store.eager,
        edges=[("habitat", "keywords")],
        start_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]
