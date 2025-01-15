import statistics
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.generic_graph_traversal_retriever import (
    GenericGraphTraversalRetriever,
)
from graph_pancake.retrievers.node import Node
from graph_pancake.retrievers.node_selectors.eager_scoring_node_selector import (
    EagerScoringNodeSelector,
)
from graph_pancake.retrievers.traversal_adapters.generic.base import (
    StoreAdapter,
)
from tests.integration_tests.retrievers.conftest import (
    doc_ids,
    supports_normalized_metadata,
)

ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]


class EagerLegScoringNodeSelector(EagerScoringNodeSelector):
    def leg_score(self, doc: Node) -> float:
        score = (
            10 - doc.metadata.get("number_of_legs", 0) - statistics.mean(doc.embedding)
        )
        print(f"{doc.id} score: {score}")
        return score

    def __init__(self, *, select_k=1, **kwargs: dict[str, Any]):
        super().__init__(self.leg_score, select_k=select_k, **kwargs)


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
        node_selector_factory=EagerLegScoringNodeSelector,
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
    assert doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)

    assert doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "mongoose",
    ]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert doc_ids(docs) == [
        "alpaca",
        "bison",
        "cat",
        "chicken",
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
        node_selector_factory=EagerLegScoringNodeSelector,
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
    assert doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)
    assert doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert doc_ids(docs) == [
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
        node_selector_factory=EagerLegScoringNodeSelector,
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
    assert doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=1)
    assert doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = retriever.invoke(ANIMALS_QUERY, max_depth=2)
    assert doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]
