from graph_retriever.adapters.base import Adapter
from graph_retriever.strategies.scored import Scored
from graph_retriever.types import Node

from tests.testing.adapters import (
    ANIMALS_QUERY,
)
from tests.testing.invoker import SyncOrAsync


def score_animals(node: Node) -> float:
    return 20 - len(node.id)


async def test_animals_keywords(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with lists."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Scored(scorer=score_animals, start_k=2),
    )

    # start_k=2 => 2 closest matches to the query
    assert await traversal(max_depth=0) == [
        "fox",
        "mongoose",
    ]
    # k=8, we start with 2 closest and choose 6 more with shortest names
    assert await traversal(k=8, max_depth=1) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]
    # k=4, we start with 2 closest and choose 2 more with shortest names
    # (from "cat", "coyote", "gazelle", "hyena", "jackal")
    assert await traversal(k=4, max_depth=1) == [
        "cat",
        "fox",
        "hyena",
        "mongoose",
    ]
