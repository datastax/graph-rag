from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from graph_pancake.retrievers.graph_traversal_retriever import (
    GraphTraversalRetriever,
)
from graph_pancake.retrievers.store_adapters.in_memory import (
    InMemoryStoreAdapter,
)
from graph_pancake.retrievers.strategy.mmr import (
    Mmr,
)
from tests.embeddings.fake_embeddings import AngularTwoDimensionalEmbeddings
from tests.integration_tests.assertions import sorted_doc_ids
from tests.integration_tests.retrievers.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.integration_tests.stores import StoreAdapter


async def test_animals_bidir_collection(animal_store: StoreAdapter, invoker):
    # test graph-search on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=["keywords"],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=4, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=4, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["cat", "gazelle", "hyena", "mongoose"]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=6, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == [
        "bison",
        "cat",
        "fox",
        "gazelle",
        "hyena",
        "mongoose",
    ]


async def test_animals_bidir_item(animal_store: StoreAdapter, invoker):
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=["habitat"],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=1)
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
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=2)
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
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Mmr(k=10, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


async def test_traversal_mem(invoker) -> None:
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
    retriever = GraphTraversalRetriever(
        store=InMemoryStoreAdapter(
            vector_store=store, support_normalized_metadata=False
        ),
        edges=[("outgoing", "incoming")],
        strategy=strategy,
    )

    docs: list[Document] = await invoker(retriever, "0.0")
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = await invoker(retriever, "0.0", strategy={"max_depth": 0})
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `start_k`, we encounter v2
    docs = await invoker(retriever, "0.0", strategy={"start_k": 3, "max_depth": 0})
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = await invoker(retriever, "0.0", strategy={"score_threshold": 0.2})
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = await invoker(retriever, "0.0", strategy={"k": 4})
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]
