import pytest
from graph_retriever.adapters.base import Adapter
from graph_retriever.strategies.scored import Scored
from graph_retriever.testing.adapter_tests import cosine_similarity_scores
from graph_retriever.types import Node

from tests.testing.adapters import ANIMALS_QUERY
from tests.testing.invoker import SyncOrAsync


def score_animals(node: Node) -> float:
    return (20 - len(node.id)) + ((ord(node.id[0]) - ord("a")) / 100)


async def test_animals_keywords(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with lists."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Scored(scorer=score_animals, start_k=2),
    )

    # # start_k=2 => 2 closest matches to the query
    assert await traversal(max_depth=0) == [
        "fox",
        "mongoose",
    ]
    # # select_k=8, we start with 2 closest and choose 6 more with shortest names
    assert await traversal(select_k=8, max_depth=1) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]
    # select_k=4, we start with 2 closest and choose 2 more with shortest names
    # (from "cat", "coyote", "gazelle", "hyena", "jackal")
    assert await traversal(select_k=4, max_depth=1) == [
        "cat",
        "fox",
        "hyena",
        "jackal",
    ]


async def test_animals_populates_metrics_and_order(
    animals: Adapter, sync_or_async: SyncOrAsync
):
    """Test that score and depth are populated and results are returned in order."""
    results = await sync_or_async.traverse(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("habitat", "habitat")],
        strategy=Scored(scorer=score_animals, start_k=2),
    )(select_k=8, max_depth=2)

    expected_scores = {
        "mongoose": 12.12,
        "bobcat": 14.01,
        "cobra": 15.02,
        "deer": 16.03,
        "elk": 17.04,
        "fox": 17.05,
    }
    expected_depths = {
        "mongoose": 0,
        "bobcat": 1,
        "cobra": 1,
        "deer": 1,
        "elk": 1,
        "fox": 0,
    }

    expected_similarity_scores = cosine_similarity_scores(
        animals, ANIMALS_QUERY, list(expected_depths.keys())
    )

    for n in results:
        assert n.extra_metadata["_similarity_score"] == pytest.approx(
            expected_similarity_scores[n.id]
        ), f"incorrect similarity score for {n.id}"
        assert n.extra_metadata["_score"] == expected_scores[n.id], (
            f"incorrect score for {n.id}"
        )
        assert n.extra_metadata["_depth"] == expected_depths[n.id], (
            f"incorrect depth for {n.id}"
        )

    expected_ids_in_order = sorted(
        expected_scores.keys(), key=lambda id: expected_scores[id], reverse=True
    )
    assert [n.id for n in results] == expected_ids_in_order, (
        "incorrect order of results"
    )
