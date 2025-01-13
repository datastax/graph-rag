import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.strategy.mmr import (
    Mmr,
)
from graph_pancake.retrievers.traversal_adapters.generic.in_memory import (
    InMemoryStoreAdapter,
)
from tests.conftest import sorted_doc_ids
from tests.embeddings.fake_embeddings import AngularTwoDimensionalEmbeddings


@pytest.fixture(scope="function", params=["norm", "denorm"])
def animal_store_adapter(
    animal_store: InMemoryVectorStore, request: pytest.FixtureRequest
) -> InMemoryStoreAdapter:
    return InMemoryStoreAdapter(
        animal_store, support_normalized_metadata=request.param == "norm"
    )


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


def test_mmr_parameters() -> None:
    mmr1 = Mmr(query_embedding=[0.25, 0.5, 0.75])
    assert id(mmr1._nd_query_embedding) == id(mmr1._nd_query_embedding)

    mmr2 = mmr1.model_copy(deep=True)
    assert id(mmr1._nd_query_embedding) != id(mmr2._nd_query_embedding)


async def test_animals_mmr_bidir_collection(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    # test graph-search on a normalized bi-directional edge
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["keywords"],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=4, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=4, start_k=2, max_depth=1))

    if not animal_store_adapter.support_normalized_metadata:
        # If we don't support normalized data, then no edges are traversed.
        assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED
        return

    assert sorted_doc_ids(docs) == ["cat", "gazelle", "jackal", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=6, start_k=2, max_depth=2))
    assert sorted_doc_ids(docs) == [
        "cat",
        "cockroach",
        "coyote",
        "gazelle",
        "jackal",
        "mongoose",
    ]


async def test_animals_mmr_bidir_item(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=["habitat"],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=1))
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=2))
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


async def test_animals_mmr_item_to_collection(
    animal_store_adapter: InMemoryStoreAdapter, invoker
):
    retriever = GenericGraphTraversalRetriever(
        store=animal_store_adapter,
        edges=[("habitat", "keywords")],
    )

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=0))
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=1))
    if not animal_store_adapter.support_normalized_metadata:
        assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED
        return

    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, Mmr(k=10, start_k=2, max_depth=2))
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


async def test_mmr_traversal(invoker) -> None:
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

    strategy = Mmr(k=2, start_k=2, max_depth=2)
    retriever = GenericGraphTraversalRetriever(
        store=InMemoryStoreAdapter(vector_store=store),
        edges=[("outgoing", "incoming")],
        strategy=strategy,
    )

    docs = await invoker(retriever, "0.0")
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = await invoker(retriever, "0.0", strategy.model_copy(update={"max_depth": 0}))
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `start_k`, we encounter v2
    docs = await invoker(
        retriever, "0.0", strategy.model_copy(update={"start_k": 3, "max_depth": 0})
    )
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = await invoker(
        retriever, "0.0", strategy.model_copy(update={"score_threshold": 0.2})
    )
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = await invoker(retriever, "0.0", strategy.model_copy(update={"k": 4}))
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]
