import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.strategy.eager import (
    Eager,
)
from graph_pancake.retrievers.traversal_adapters.generic.in_memory import (
    InMemoryStoreAdapter,
)
from tests.conftest import assert_document_format, sorted_doc_ids
from tests.embeddings.simple_embeddings import ParserEmbeddings


@pytest.fixture(scope="function", params=["norm", "denorm"])
def animal_store_adapter(
    animal_store: InMemoryVectorStore, request: pytest.FixtureRequest
) -> InMemoryStoreAdapter:
    return InMemoryStoreAdapter(
        animal_store, support_normalized_metadata=request.param == "norm"
    )


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


async def test_animals_bidir_collection_eager(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    # test graph-search on a normalized bi-directional edge
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["keywords"],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=100, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=100, start_k=2, max_depth=1))

    if not animal_store_adapter.support_normalized_metadata:
        # If we don't support normalized data, then no edges are traversed.
        assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED
        return

    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=100, start_k=2, max_depth=2))
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


async def test_animals_eager_bidir_item(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["habitat"],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=1))
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=2))
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


async def test_animals_eager_item_to_collection(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=[("habitat", "keywords")],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=1))
    if not animal_store_adapter.support_normalized_metadata:
        assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED
        return

    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, Eager(k=10, start_k=2, max_depth=2))
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


@pytest.fixture(scope="function")
def parser_adapter(graph_vector_store_docs: list[Document]) -> InMemoryStoreAdapter:
    store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
    store.add_documents(graph_vector_store_docs)
    return InMemoryStoreAdapter(store)


async def test_parser_eager(parser_adapter: InMemoryStoreAdapter, invoker):
    retriever = GenericGraphTraversalRetriever(
        store=parser_adapter,
        edges=[("out", "in"), "tag"],
        strategy=Eager(k=10, start_k=2, max_depth=2),
    )

    docs = await invoker(retriever, "[2, 10]", Eager(k=10, start_k=2, max_depth=0))
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await invoker(retriever, "[2, 10]")
    # this is a set, as some of the internals of trav.search are set-driven
    # so ordering is not deterministic:
    ts_labels = {doc.metadata["label"] for doc in docs}
    assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])
