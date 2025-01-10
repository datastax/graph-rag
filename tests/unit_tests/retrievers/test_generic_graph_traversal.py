import json
from typing import Iterable

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.node_selectors.eager_node_selector import (
    EagerNodeSelector,
)
from graph_pancake.retrievers.node_selectors.mmr_scoring_node_selector import (
    MmrScoringNodeSelector,
)
from graph_pancake.retrievers.traversal_adapters.generic.in_memory import (
    InMemoryStoreAdapter,
)
from tests.embeddings.fake_embeddings import AngularTwoDimensionalEmbeddings
from tests.embeddings.simple_embeddings import AnimalEmbeddings, ParserEmbeddings


def sorted_doc_ids(docs: Iterable[Document]) -> list[str]:
    return sorted([doc.id for doc in docs if doc.id is not None])


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


@pytest.fixture(scope="module")
def animal_docs() -> list[Document]:
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


@pytest.fixture(scope="function", params=[False, True])
def animal_store_adapter(
    animal_docs: list[Document], request: pytest.FixtureRequest
) -> InMemoryStoreAdapter:
    store = InMemoryVectorStore(embedding=AnimalEmbeddings())
    store.add_documents(animal_docs)

    return InMemoryStoreAdapter(store, support_normalized_metadata=request.param)


@pytest.fixture(scope="function")
def parser_adapter(graph_vector_store_docs: list[Document]) -> InMemoryStoreAdapter:
    store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
    store.add_documents(graph_vector_store_docs)

    return InMemoryStoreAdapter(store)


def test_parser_eager_sync(parser_adapter: InMemoryStoreAdapter):
    retriever = GenericGraphTraversalRetriever(
        store=parser_adapter,
        edges=[("out", "in"), "tag"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        extra_args={
            "max_depth": 2
        },
    )

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


async def test_parser_eager_async(parser_adapter: InMemoryStoreAdapter):
    retriever = GenericGraphTraversalRetriever(
        store=parser_adapter,
        edges=[("out", "in"), "tag"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
        extra_args={
            "max_depth": 2
        },
    )
    docs = await retriever.ainvoke(input="[2, 10]", max_depth=0)
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await retriever.ainvoke(input="[2, 10]")
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


def test_animals_bidir_collection_eager(animal_store_adapter: InMemoryStoreAdapter):
    # test graph-search on a normalized bi-directional edge
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["keywords"],
        node_selector_factory=EagerNodeSelector,
        k=100,
        start_k=2,
    )

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)

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


def test_animals_eager_bidir_item(animal_store_adapter: InMemoryStoreAdapter):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["habitat"],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
    )

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


def test_animals_eager_item_to_collection(animal_store_adapter: InMemoryStoreAdapter):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=[("habitat", "keywords")],
        node_selector_factory=EagerNodeSelector,
        k=10,
        start_k=2,
    )

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)
    if not animal_store_adapter.support_normalized_metadata:
        assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED
        return

    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


def test_mmr_traversal() -> None:
    """ Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          //      \\
         //        \\  v1
    v3  ||    .     || query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With start_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    store = InMemoryVectorStore(embedding=AngularTwoDimensionalEmbeddings())
    store.add_documents([v0, v1, v2, v3])

    retriever = GenericGraphTraversalRetriever(
        store=InMemoryStoreAdapter(vector_store=store),
        edges=[("outgoing", "incoming")],
        node_selector_factory=MmrScoringNodeSelector,
        start_k=2,
        k=2,
        extra_args={
            "max_depth": 2
        },
    )

    # docs = retriever.invoke("0.0", k=2, start_k=2)
    # assert sorted_doc_ids(docs) == ["v0", "v2"]

    # # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # # So it ends up picking "v1" even though it's similar to "v0".
    # docs = retriever.invoke("0.0", k=2, start_k=2, max_depth=0)
    # assert sorted_doc_ids(docs) == ["v0", "v1"]

    # # With max depth 0 but higher `start_k`, we encounter v2
    # docs = retriever.invoke("0.0", k=2, start_k=3, max_depth=0)
    # assert sorted_doc_ids(docs) == ["v0", "v2"]

    # # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    # docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    # assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]
