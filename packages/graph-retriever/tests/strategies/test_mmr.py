import pytest
from graph_retriever.adapters.base import Adapter
from graph_retriever.adapters.in_memory import InMemory
from graph_retriever.content import Content
from graph_retriever.strategies.mmr import Mmr
from graph_retriever.testing.embeddings import angular_2d_embedding

from tests.testing.adapters import ANIMALS_DEPTH_0_EXPECTED, ANIMALS_QUERY
from tests.testing.invoker import SyncOrAsync


async def test_animals_keywords(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with lists."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Mmr(start_k=2),
    )

    assert await traversal(select_k=4, max_depth=0) == ANIMALS_DEPTH_0_EXPECTED
    assert await traversal(select_k=4, max_depth=1) == [
        "cat",
        "fox",
        "gazelle",
        "mongoose",
    ]
    assert await traversal(select_k=6, max_depth=2) == [
        "cat",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]


async def test_rediscovering(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test for https://github.com/datastax/graph-rag/issues/167.

    The issue was nodes were being "rediscovered" and readded to the candidates
    list in MMR. This violates the contract of the traversal. This test runs MMR
    with a high number of iterations (select 97 nodes, 1 at a time) and a high
    adjacent K (100) nodes at each iteration. This ensures that some nodes will
    be rediscovered.
    """
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        edges=[("habitat", "habitat")],
    )
    result = await traversal(
        query="cat",
        strategy=Mmr(select_k=97, adjacent_k=100, start_k=100, lambda_mult=0.9),
    )
    assert len(result) == 97


async def test_animals_habitat(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with singular values."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("habitat", "habitat")],
        strategy=Mmr(select_k=10, start_k=2),
    )

    assert await traversal(max_depth=0) == ANIMALS_DEPTH_0_EXPECTED
    assert await traversal(max_depth=1) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]
    assert await traversal(max_depth=2) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


async def test_animals_populates_metrics(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test that score and depth are populated."""
    results = await sync_or_async.traverse(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("habitat", "habitat")],
        strategy=Mmr(start_k=2),
    )(select_k=10, max_depth=2)

    expected_similarity_scores = {
        "mongoose": 0.578682,
        "bobcat": 0.02297939,
        "cobra": 0.01365448699,
        "deer": 0.1869947,
        "elk": 0.02876833,
        "fox": 0.533316,
    }
    expected_mmr_scores = {
        "mongoose": 0.28934083735912275,
        "fox": 0.11235363166682244,
        "deer": 0.03904356616509902,
        "bobcat": 0.0031420490138288626,
        "cobra": -0.11165876337613051,
        "elk": -0.22759302101291784,
    }
    expected_redundancy = {
        "mongoose": 0.0,
        "fox": 0.30860872491035307,
        "deer": 0.10890754955985982,
        "bobcat": 0.016695295174737335,
        "cobra": 0.24437328139277636,
        "elk": 0.4839543733764524,
    }
    expected_depths = {
        "mongoose": 0,
        "bobcat": 1,
        "cobra": 1,
        "deer": 1,
        "elk": 1,
        "fox": 0,
    }

    for n in results:
        assert n.extra_metadata["_similarity_score"] == pytest.approx(
            expected_similarity_scores[n.id]
        ), f"incorrect similarity score for {n.id}"
        assert n.extra_metadata["_mmr_score"] == pytest.approx(
            expected_mmr_scores[n.id]
        ), f"incorrect score for {n.id}"
        assert n.extra_metadata["_redundancy"] == pytest.approx(
            expected_redundancy[n.id]
        ), f"incorrect redundancy for {n.id}"
        assert n.extra_metadata["_depth"] == expected_depths[n.id], (
            f"incorrect depth for {n.id}"
        )


async def test_animals_habitat_to_keywords(
    animals: Adapter, sync_or_async: SyncOrAsync
):
    """Test traversing a from a singular field (habitat) to collection (keywords)."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("habitat", "keywords")],
        strategy=Mmr(select_k=10, start_k=2),
    )

    assert await traversal(max_depth=0) == ANIMALS_DEPTH_0_EXPECTED
    assert await traversal(max_depth=1) == ["bear", "bobcat", "fox", "mongoose"]
    assert await traversal(max_depth=2) == [
        "bear",
        "bobcat",
        "caribou",
        "fox",
        "mongoose",
    ]


async def test_angular(sync_or_async: SyncOrAsync):
    """
    Test end to end construction and MMR search.

    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          //      \\
         //        \\  v1
    v3  ||    .     || query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With start_k==2 and select_k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    embedding = angular_2d_embedding
    v0 = Content.new("v0", "-0.124", embedding, metadata={"outgoing": "link"})
    v1 = Content.new("v1", "+0.127", embedding)
    v2 = Content.new("v2", "+0.25", embedding, metadata={"incoming": "link"})
    v3 = Content.new("v3", "+1.0", embedding, metadata={"incoming": "link"})

    traversal = sync_or_async.traverse_sorted_ids(
        query="0.0",
        store=InMemory(embedding, [v0, v1, v2, v3]),
        edges=[("outgoing", "incoming")],
        strategy=Mmr(select_k=2, start_k=2, max_depth=2),
    )

    assert await traversal() == ["v0", "v2"]
    assert await traversal() == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    assert await traversal(max_depth=0) == ["v0", "v1"]

    # With max depth 0 but higher `start_k`, we encounter v2
    assert await traversal(start_k=3, max_depth=0) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    assert await traversal(min_mmr_score=0.2) == ["v0"]

    # with select_k=4 we should get all of the documents.
    assert await traversal(select_k=4) == ["v0", "v1", "v2", "v3"]
