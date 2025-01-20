
from langchain_core.documents import Document
from langchain_graph_retriever import (
    GraphRetriever,
)
from langchain_graph_retriever.node import Node
from langchain_graph_retriever.adapters import (
    Adapter,
)
from langchain_graph_retriever.strategies import (
    Scored,
)
from tests.integration_tests.assertions import doc_ids
from tests.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)


class AnimalScore(Scored):
    # score animals based on number of weight, legs, and tail-length
    def test(self, doc: Node) -> float:
        legs = doc.metadata.get("number_of_legs", 0)
        weight = doc.metadata.get("weight", 0)
        tail_length = doc.metadata.get("tail_length", 0)

        score = (1000 * weight) + legs + tail_length


        return score

    def __init__(self, **kwargs):
        super().__init__(scorer=self.test, select_k=2, **kwargs)



async def test_animals_bidir_collection(animal_store: Adapter, invoker):
    # test graph-search on a normalized bi-directional edge
    retriever = GraphRetriever(
        store=animal_store,
        edges=["keywords"],
    )
    print("\n1")
    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(start_k=3, max_depth=0)
    )

    # maybe auto includes start_k nodes before starting the section_k selections
    assert doc_ids(docs) == ["hedgehog", "mongoose", "fox"]


    print("\n2")
    # select_k 2, start_k2, k=6, depth=1
    # add 2 via start_k, select 2 via select_k, traverse, find 5 more, select using select_k (in groups of select_k), until k is reached
    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=AnimalScore(k=6, start_k=2, max_depth=1)
    )
    assert doc_ids(docs) == ['mongoose', 'fox', 'cat', 'jackal', 'coyote', 'gazelle']

    # select_k=2, start_k=2, k=6, depth=2
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


async def test_animals_bidir_item(animal_store: Adapter, invoker):
    retriever = GraphRetriever(
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


async def test_animals_item_to_collection(animal_store: Adapter, invoker):
    retriever = GraphRetriever(
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
