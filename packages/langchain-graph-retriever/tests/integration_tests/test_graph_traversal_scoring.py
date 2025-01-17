import statistics
from typing import Any

from langchain_core.documents import Document

from graph_pancake.retrievers.graph_traversal_retriever import (
    GraphTraversalRetriever,
)
from graph_pancake.retrievers.node import Node
from graph_pancake.retrievers.strategy.scored import (
    Scored,
)
from graph_pancake.retrievers.store_adapters import (
    StoreAdapter,
)


from tests.integration_tests.retrievers.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.integration_tests.assertions import doc_ids

class AnimalScore(Scored):

    # score animals based on number of weight, legs, and tail-length
    def test(self, doc: Node) -> float:
        legs = doc.metadata.get("number_of_legs", 0)
        weight = doc.metadata.get("weight", 0)
        tail_length = doc.metadata.get("tail_length", 0)

        score = (1000 * weight) + legs + tail_length

        # score = (
        #     10 -  - statistics.mean(doc.embedding)
        # )
        return score

    def __init__(self, **kwargs):
        super().__init__(scorer=self.test, select_k=2, **kwargs)

async def test_animals_bidir_collection(animal_store: StoreAdapter, invoker):
    # test graph-search on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=animal_store,
        edges=["keywords"],
    )
    print("\n1")
    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(start_k=3, max_depth=0)
    )
    assert doc_ids(docs) == ['hedgehog', 'mongoose', 'fox']
    print("\n2")
    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=6, start_k=2, max_depth=1)
    )
    assert doc_ids(docs) == ['mongoose', 'fox', 'cat', 'jackal']
    print("\n3")
    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=6, start_k=2, max_depth=2)
    )
    assert doc_ids(docs) == [
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
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=0)
    )
    assert doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=1)
    )
    assert doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=2)
    )
    assert doc_ids(docs) == [
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
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=0)
    )
    assert doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=1)
    )
    assert doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=10, start_k=2, max_depth=2)
    )
    assert doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]
