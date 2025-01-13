"""Test of MMR Graph Traversal Retriever"""

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.graph_mmr_traversal_retriever import (
    GraphMMRTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.mmr import (
    AstraMMRTraversalAdapter,
    CassandraMMRTraversalAdapter,
    ChromaMMRTraversalAdapter,
    MMRTraversalAdapter,
    OpenSearchMMRTraversalAdapter,
)
from tests.conftest import (
    assert_document_format,
    sorted_doc_ids,
    supports_normalized_metadata,
)

vector_store_types = [
    #   "astra-db",
    "cassandra",
    "chroma-db",
    "open-search",
]


def get_adapter(
    vector_store: VectorStore, vector_store_type: str
) -> MMRTraversalAdapter:
    if vector_store_type == "astra-db":
        return AstraMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "cassandra":
        return CassandraMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "chroma-db":
        return ChromaMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "open-search":
        return OpenSearchMMRTraversalAdapter(vector_store=vector_store)
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["angular"])
def test_mmr_traversal(
    vector_store: VectorStore, vector_store_type: str, mmr_docs: list[Document]
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
    if not supports_normalized_metadata(vector_store_type=vector_store_type):
        mmr_docs = MetadataDenormalizer().transform_documents(mmr_docs)

    vector_store.add_documents(mmr_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        edges=[("outgoing", "incoming")],
        fetch_k=2,
        k=2,
        depth=2,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            docs = retriever.invoke("0.0", k=2, fetch_k=2)
        return

    docs = retriever.invoke("0.0", k=2, fetch_k=2)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = retriever.invoke("0.0", k=2, fetch_k=2, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    docs = retriever.invoke("0.0", k=2, fetch_k=3, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["parser-d2"])
def test_invoke_sync(
    vector_store: VectorStore,
    vector_store_type: str,
    graph_vector_store_docs: list[Document],
) -> None:
    """MMR Graph traversal search on a vector store."""
    vector_store.add_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        vector_store=vector_store,
        edges=[("out", "in"), "tag"],
        depth=2,
        k=2,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            docs = retriever.invoke(input="[2, 10]")
        return

    docs = retriever.invoke(input="[2, 10]")
    mt_labels = {doc.metadata["label"] for doc in docs}
    assert mt_labels == {"AR", "BR"}
    assert docs[0].metadata
    assert_document_format(docs[0])


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["parser-d2"])
async def test_invoke_async(
    vector_store: VectorStore,
    vector_store_type: str,
    graph_vector_store_docs: list[Document],
) -> None:
    """MMR Graph traversal search on a vector store."""
    await vector_store.aadd_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        vector_store=vector_store,
        edges=[("out", "in"), "tag"],
        depth=2,
        k=2,
    )
    mt_labels = set()
    docs = await retriever.ainvoke(input="[2, 10]")
    mt_labels = {doc.metadata["label"] for doc in docs}
    assert mt_labels == {"AR", "BR"}
    assert docs[0].metadata
    assert_document_format(docs[0])


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["animal"])
def test_animals_sync(
    vector_store: VectorStore,
    vector_store_type: str,
    animal_docs: list[Document],
) -> None:
    use_denormalized_metadata = not supports_normalized_metadata(
        vector_store_type=vector_store_type
    )

    if use_denormalized_metadata:
        animal_docs = MetadataDenormalizer().transform_documents(animal_docs)

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
    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        edges=["keywords"],
        fetch_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    if vector_store_type == "cassandra":
        with pytest.raises(
            NotImplementedError, match="use the async implementation instead"
        ):
            docs = retriever.invoke(query)
        return

    docs = retriever.invoke(query, depth=0)
    assert sorted_doc_ids(docs) == depth_0_expected

    docs = retriever.invoke(query, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = retriever.invoke(query, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["cat", "gazelle", "jackal", "mongoose"]

    # test graph-search on a standard bi-directional edge
    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        edges=["habitat"],
        fetch_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = retriever.invoke(query, depth=0)
    assert sorted_doc_ids(docs) == depth_0_expected

    docs = retriever.invoke(query, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = retriever.invoke(query, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["bobcat", "deer", "fox", "mongoose"]

    # test graph-search on a standard -> normalized edge
    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        edges=[("habitat", "keywords")],
        fetch_k=2,
        use_denormalized_metadata=use_denormalized_metadata,
    )

    docs = retriever.invoke(query, depth=0)
    assert sorted_doc_ids(docs) == depth_0_expected

    docs = retriever.invoke(query, depth=1)
    assert sorted_doc_ids(docs) == ["fox", "mongoose"]

    docs = retriever.invoke(query, depth=2)
    # WOULD HAVE EXPECTED THIS AT DEPTH 1
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]
