"""Test of Graph Traversal Retriever"""

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.graph_traversal_retriever import GraphTraversalRetriever
from graph_pancake.retrievers.traversal_adapters.eager import (
    AstraTraversalAdapter,
    CassandraTraversalAdapter,
    ChromaTraversalAdapter,
    InMemoryTraversalAdapter,
    OpenSearchTraversalAdapter,
    TraversalAdapter,
)
from tests.integration_tests.retrievers.conftest import (
    assert_document_format,
    sorted_doc_ids,
    supports_normalized_metadata,
)


def get_adapter(vector_store: VectorStore, vector_store_type: str) -> TraversalAdapter:
    if vector_store_type == "astra-db":
        return AstraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "cassandra":
        return CassandraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "chroma-db":
        return ChromaTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "open-search":
        return OpenSearchTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "in-memory":
        return InMemoryTraversalAdapter(
            vector_store=vector_store, support_normalized_metadata=True
        )
    elif vector_store_type == "in-memory-denormalized":
        return InMemoryTraversalAdapter(
            vector_store=vector_store, support_normalized_metadata=False
        )
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


@pytest.mark.parametrize("embedding_type", ["earth"])
def test_traversal(
    vector_store_type: str,
    vector_store: VectorStore,
    hello_docs: list[Document],
) -> None:
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        hello_docs = list(MetadataDenormalizer().transform_documents(hello_docs))

    vector_store.add_documents(hello_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
        edges=[("outgoing", "incoming"), "keywords"],
        start_k=2,
        depth=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = retriever.invoke("Earth", start_k=1, depth=0)
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = retriever.invoke("Earth", depth=0)
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = retriever.invoke("Earth", start_k=1, depth=1)
    assert sorted_doc_ids(docs) == ["doc1", "doc2", "greetings"]


@pytest.mark.parametrize("embedding_type", ["parser-d2"])
def test_invoke_sync(
    vector_store_type: str,
    vector_store: VectorStore,
    graph_vector_store_docs: list[Document],
) -> None:
    """Graph traversal search on a vector store."""
    vector_store.add_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
        edges=[("out", "in"), "tag"],
        depth=2,
        start_k=2,
    )

    docs = retriever.invoke(input="[2, 10]", depth=0)
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = retriever.invoke(input="[2, 10]")
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


@pytest.mark.parametrize("embedding_type", ["parser-d2"])
async def test_invoke_async(
    vector_store_type: str,
    vector_store: VectorStore,
    graph_vector_store_docs: list[Document],
) -> None:
    """Graph traversal search on a graph store."""
    await vector_store.aadd_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
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


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_sync(
    vector_store_type: str,
    vector_store: VectorStore,
    animal_docs: list[Document],
) -> None:
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    vector_store.add_documents(animal_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    query = "small agile mammal"
    depth_0_expected = ["fox", "mongoose"]

    # test non-graph search
    docs = vector_store.similarity_search(query, k=2)
    assert sorted_doc_ids(docs) == depth_0_expected

    # test graph-search on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
        edges=["keywords"],
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = retriever.invoke(query, depth=0)
    assert sorted_doc_ids(docs) == depth_0_expected

    docs = retriever.invoke(query, depth=1)
    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = retriever.invoke(query, depth=2)
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
        store=vector_store_adapter,
        edges=["habitat"],
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
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
        store=vector_store_adapter,
        edges=[("habitat", "keywords")],
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = retriever.invoke(query, depth=0)
    assert sorted_doc_ids(docs) == depth_0_expected

    docs = retriever.invoke(query, depth=1)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = retriever.invoke(query, depth=2)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]
