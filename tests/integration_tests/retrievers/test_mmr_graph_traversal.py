"""Test of MMR Graph Traversal Retriever"""

import pytest
from langchain_core.documents import Document

from graph_pancake.retrievers.graph_mmr_traversal_retriever import (
    GraphMMRTraversalRetriever,
)
from tests.embeddings.fake_embeddings import AngularTwoDimensionalEmbeddings
from tests.integration_tests.assertions import (
    assert_document_format,
    sorted_doc_ids,
)
from tests.integration_tests.retrievers.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.integration_tests.stores import StoreFactory, Stores


async def test_traversal(
    request: pytest.FixtureRequest, store_factory: StoreFactory, invoker
) -> None:
    """Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          //      \\
         //        \\  v1
    v3  ||    .     || query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
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

    store = store_factory.create(
        request, AngularTwoDimensionalEmbeddings(), docs=[v0, v1, v2, v3]
    )

    retriever = GraphMMRTraversalRetriever(
        store=store.mmr,
        edges=[("outgoing", "incoming")],
        fetch_k=2,
        k=2,
        depth=2,
    )

    docs: list[Document] = await invoker(retriever, "0.0", k=2, fetch_k=2)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = await invoker(retriever, "0.0", k=2, fetch_k=2, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    docs = await invoker(retriever, "0.0", k=2, fetch_k=3, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = await invoker(retriever, "0.0", k=2, score_threshold=0.2)
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = await invoker(retriever, "0.0", k=4)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]


async def test_invoke(parser_store: Stores, invoker) -> None:
    """MMR Graph traversal search on a vector store."""
    retriever = GraphMMRTraversalRetriever(
        store=parser_store.mmr,
        edges=[("out", "in"), "tag"],
        depth=2,
        k=2,
    )

    docs: list[Document] = await invoker(retriever, input="[2, 10]")
    mt_labels = {doc.metadata["label"] for doc in docs}
    assert mt_labels == {"AR", "BR"}
    assert docs[0].metadata
    assert_document_format(docs[0])


async def test_animals(
    animal_store: Stores, support_normalized_metadata: bool, invoker
) -> None:
    # test graph-search on a normalized bi-directional edge
    retriever = GraphMMRTraversalRetriever(
        store=animal_store.mmr,
        edges=["keywords"],
        fetch_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs: list[Document] = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["cat", "gazelle", "hyena", "mongoose"]

    # test graph-search on a standard bi-directional edge
    retriever = GraphMMRTraversalRetriever(
        store=animal_store.mmr,
        edges=["habitat"],
        fetch_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["bobcat", "deer", "fox", "mongoose"]

    # test graph-search on a standard -> normalized edge
    retriever = GraphMMRTraversalRetriever(
        store=animal_store.mmr,
        edges=[("habitat", "keywords")],
        fetch_k=2,
        use_denormalized_metadata=not support_normalized_metadata,
    )

    docs = await invoker(retriever, ANIMALS_QUERY, depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = await invoker(retriever, ANIMALS_QUERY, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]
