import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.node_selectors.eager_node_selector import (
    EagerNodeSelector,
)
from graph_pancake.retrievers.traversal_adapters.generic.base import (
    StoreAdapter,
)
from tests.integration_tests.retrievers.conftest import (
    assert_document_format,
    sorted_doc_ids,
    supports_normalized_metadata,
)

ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_eager_bidir_collection(
    vector_store_type: str,
    vector_store: VectorStore,
    animal_docs: list[Document],
    store_adapter: StoreAdapter,
):
    # test graph-search on a normalized bi-directional edge
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    vector_store.add_documents(animal_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=["keywords"],
        node_selector_factory=EagerNodeSelector,
        k=100,
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            retriever.invoke(ANIMALS_QUERY)
        return

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)

    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
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


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_eager_bidir_item(
    vector_store_type: str,
    vector_store: VectorStore,
    animal_docs: list[Document],
    store_adapter: StoreAdapter,
):
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    vector_store.add_documents(animal_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=["habitat"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            retriever.invoke(ANIMALS_QUERY)
        return

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_eager_item_to_collection(
    vector_store_type: str,
    vector_store: VectorStore,
    animal_docs: list[Document],
    store_adapter: StoreAdapter,
):
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    vector_store.add_documents(animal_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=[("habitat", "keywords")],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            retriever.invoke(ANIMALS_QUERY)
        return

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


@pytest.mark.parametrize("embedding_type", ["parser-d2"])
def test_parser_eager_sync(
    vector_store_type: str,
    vector_store: VectorStore,
    graph_vector_store_docs: list[Document],
    store_adapter: StoreAdapter,
):
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        graph_vector_store_docs = list(
            MetadataDenormalizer().transform_documents(graph_vector_store_docs)
        )

    vector_store.add_documents(graph_vector_store_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=[("out", "in"), "tag"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        extra_args={"max_depth": 2},
        use_denormalized_metadata=use_denormalized_metadata,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            retriever.invoke(input="[2, 10]", max_depth=0)
        return

    docs = retriever.invoke(input="[2, 10]", max_depth=0)
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = retriever.invoke(input="[2, 10]")
    # this is a set, as some of the internals of trav.search are set-driven
    # so ordering is not deterministic:
    ts_labels = {doc.metadata["label"] for doc in docs}
    assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


@pytest.mark.parametrize("embedding_type", ["parser-d2"])
async def test_parser_eager_async(
    vector_store_type: str,
    vector_store: VectorStore,
    graph_vector_store_docs: list[Document],
    store_adapter: StoreAdapter,
):
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        graph_vector_store_docs = list(
            MetadataDenormalizer().transform_documents(graph_vector_store_docs)
        )

    await vector_store.aadd_documents(graph_vector_store_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=[("out", "in"), "tag"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        extra_args={"max_depth": 2},
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = await retriever.ainvoke(input="[2, 10]", max_depth=0)
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await retriever.ainvoke(input="[2, 10]")
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])
