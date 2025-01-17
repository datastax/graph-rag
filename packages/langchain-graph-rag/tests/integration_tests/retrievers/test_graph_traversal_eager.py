from langchain_core.documents import Document
from langchain_graph_rag.retrievers.graph_traversal_retriever import (
    GraphTraversalRetriever,
)
from langchain_graph_rag.retrievers.strategy.eager import (
    Eager,
)
from tests.integration_tests.assertions import assert_document_format, sorted_doc_ids
from tests.integration_tests.retrievers.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.integration_tests.stores import StoreAdapter


async def test_animals_bidir_collection_eager(animal_store: StoreAdapter, invoker):
    # test graph-search on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=["keywords"],
        strategy=Eager(k=100, start_k=2, max_depth=0),
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy={"max_depth": 0}
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, strategy={"max_depth": 1})
    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, strategy={"max_depth": 2})
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


async def test_animals_bidir_item(animal_store: StoreAdapter, invoker):
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=["habitat"],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


async def test_animals_item_to_collection(animal_store: StoreAdapter, invoker):
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=[("habitat", "keywords")],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


async def test_parser(parser_store: StoreAdapter, invoker):
    retriever = GraphTraversalRetriever(
        store=parser_store,
        edges=[("out", "in"), "tag"],
        strategy=Eager(k=10, start_k=2, max_depth=2),
    )

    docs: list[Document] = await invoker(
        retriever, "[2, 10]", strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await invoker(retriever, "[2, 10]")
    # this is a set, as some of the internals of trav.search are set-driven
    # so ordering is not deterministic:
    ts_labels = {doc.metadata["label"] for doc in docs}
    assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


async def test_earth(earth_store: StoreAdapter, invoker):
    retriever = GraphTraversalRetriever(
        store=earth_store,
        edges=[("outgoing", "incoming"), "keywords"],
        strategy=Eager(k=10, start_k=2, max_depth=0),
    )

    docs: list[Document] = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=1, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=1, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["doc1", "doc2", "greetings"]
