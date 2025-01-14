import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.node_selectors.mmr_scoring_node_selector import (
    MmrScoringNodeSelector,
)
from graph_pancake.retrievers.traversal_adapters.generic.base import (
    StoreAdapter,
)
from tests.integration_tests.retrievers.conftest import (
    sorted_doc_ids,
    supports_normalized_metadata,
)

ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_mmr_bidir_collection(
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
        node_selector_factory=MmrScoringNodeSelector,
        k=4,
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
    assert sorted_doc_ids(docs) == ["cat", "gazelle", "hyena", "mongoose"]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2, k=6)
    assert sorted_doc_ids(docs) == ["bison", "cat", "fox", "gazelle", "hyena", "mongoose"]


@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_mmr_bidir_item(
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
        node_selector_factory=MmrScoringNodeSelector,
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
def test_animals_mmr_item_to_collection(
    vector_store_type: str,
    vector_store: VectorStore,
    animal_docs: list[Document],
    store_adapter: StoreAdapter,
) -> None:
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    vector_store.add_documents(animal_docs)

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=[("habitat", "keywords")],
        node_selector_factory=MmrScoringNodeSelector,
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


@pytest.mark.parametrize("embedding_type", ["angular"])
def test_mmr_traversal(
    vector_store_type: str,
    vector_store: VectorStore,
    store_adapter: StoreAdapter,
) -> None:
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

    vector_store.add_documents([v0, v1, v2, v3])

    retriever = GenericGraphTraversalRetriever(
        store=store_adapter,
        edges=[("outgoing", "incoming")],
        node_selector_factory=MmrScoringNodeSelector,
        start_k=2,
        k=2,
        extra_args={"max_depth": 2},
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            retriever.invoke("0.0")
        return

    docs = retriever.invoke("0.0", k=2, start_k=2)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = retriever.invoke("0.0", k=2, start_k=2, max_depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `start_k`, we encounter v2
    docs = retriever.invoke("0.0", k=2, start_k=3, max_depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]
