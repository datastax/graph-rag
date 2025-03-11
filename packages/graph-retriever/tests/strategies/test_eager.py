import pytest
from graph_retriever import Content
from graph_retriever.adapters import Adapter
from graph_retriever.adapters.in_memory import InMemory
from graph_retriever.edges import Edges, MetadataEdge
from graph_retriever.strategies import (
    Eager,
)
from graph_retriever.testing.embeddings import (
    ParserEmbeddings,
    angular_2d_embedding,
    earth_embeddings,
)

from tests.testing.adapters import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.testing.invoker import SyncOrAsync


async def test_earth(sync_or_async: SyncOrAsync):
    embedding = earth_embeddings
    greetings = Content(
        id="greetings",
        content="Typical Greetings",
        embedding=embedding("Typical Greetings"),
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Content(
        id="doc1",
        content="Hello World",
        embedding=embedding("Hello World"),
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Content(
        id="doc2",
        content="Hello Earth",
        embedding=embedding("Hello Earth"),
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )

    store = InMemory(embedding, [greetings, doc1, doc2])

    traversal = sync_or_async.traverse_sorted_ids(
        store=store,
        query="Earth",
        edges=[("outgoing", "incoming"), ("keywords", "keywords")],
        strategy=Eager(select_k=10),
    )

    assert await traversal(start_k=1, max_depth=0) == ["doc2"]
    assert await traversal(start_k=2, max_depth=0) == ["doc1", "doc2"]
    assert await traversal(start_k=1, max_depth=1) == ["doc1", "doc2", "greetings"]


async def test_animals_select_k(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with lists."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Eager(),
    )
    assert len(await traversal(select_k=5)) == 5
    assert len(await traversal(select_k=3)) == 3


async def test_animals_keywords(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with lists."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Eager(select_k=100, start_k=2),
    )

    assert await traversal(max_depth=0) == ANIMALS_DEPTH_0_EXPECTED
    assert await traversal(max_depth=1) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]
    assert await traversal(max_depth=2) == [
        "alpaca",
        "bison",
        "cat",
        "coyote",
        "crow",
        "dog",
        "fox",
        "gazelle",
        "horse",
        "hyena",
        "jackal",
        "mongoose",
    ]


async def test_animals_habitat(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing a bi-directional field with singular values."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("habitat", "habitat")],
        strategy=Eager(select_k=100, start_k=2),
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
        strategy=Eager(select_k=100, start_k=2, max_depth=2),
    )()

    expected_similarity_scores = {
        "mongoose": 0.578682,
        "bobcat": 0.02297939,
        "cobra": 0.01365448699,
        "deer": 0.1869947,
        "elk": 0.02876833,
        "fox": 0.533316,
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
        strategy=Eager(select_k=10, start_k=2),
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


async def test_animals_initial_roots(animals: Adapter, sync_or_async: SyncOrAsync):
    """Test traversing with initial root IDs."""
    traversal = sync_or_async.traverse_sorted_ids(
        store=animals,
        query=ANIMALS_QUERY,
        edges=[("keywords", "keywords")],
        strategy=Eager(select_k=10, start_k=0),
    )

    assert await traversal(initial_root_ids=["bobcat"], max_depth=0) == [
        "bobcat",
    ]
    assert await traversal(initial_root_ids=["bobcat"], max_depth=1) == [
        "bear",
        "bobcat",
    ]
    assert await traversal(initial_root_ids=["bobcat"], max_depth=2) == [
        "bear",
        "bobcat",
        "moose",
        "ostrich",
    ]
    assert await traversal(
        initial_root_ids=["bobcat", "cheetah"], select_k=20, max_depth=2
    ) == [
        "bear",
        "bobcat",
        "cassowary",
        "cheetah",
        "dingo",
        "eagle",
        "emu",
        "falcon",
        "hawk",
        "jaguar",
        "kangaroo",
        "leopard",
        "moose",
    ]


async def test_parsed(sync_or_async: SyncOrAsync):
    """
    This is a test of set of Documents to pre-populate,
    a graph vector store with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :
        :
        v              |  .           v
                       |   :
       TR              |   :
    T0   --------------x--------------   B0
       TL              |   :
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """
    embedding = ParserEmbeddings(2)
    docs_a = [
        Content.new("AL", "[-1, 9]", embedding),
        Content.new("A0", "[0, 10]", embedding),
        Content.new("AR", "[1, 9]", embedding),
    ]
    docs_b = [
        Content.new("BL", "[9, 1]", [9.0, 1.0]),
        Content.new("B0", "[10, 0]", [10.0, 0.0]),
        Content.new("BR", "[9, -1]", [9.0, -1.0]),
    ]
    docs_f = [
        Content.new("FL", "[1, -9]", [1.0, -9.0]),
        Content.new("F0", "[0, -10]", [0.0, -10.0]),
        Content.new("FR", "[-1, -9]", [-1.0, -9.0]),
    ]
    docs_t = [
        Content.new("TL", "[-9, -1]", [-9.0, -1.0]),
        Content.new("T0", "[-10, 0]", [-10.0, 0.0]),
        Content.new("TR", "[-9, 1]", [-9.0, 1.0]),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        doc_a.metadata["tag"] = f"ab_{suffix}"
        doc_a.metadata["out"] = f"at_{suffix}"
        doc_a.metadata["in"] = f"af_{suffix}"
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        doc_b.metadata["tag"] = f"ab_{suffix}"
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        doc_t.metadata["in"] = f"at_{suffix}"
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        doc_f.metadata["out"] = f"af_{suffix}"
    documents = docs_a + docs_b + docs_f + docs_t

    traversal = sync_or_async.traverse_sorted_ids(
        store=InMemory(embedding, documents),
        edges=[("out", "in"), ("tag", "tag")],
        query="[2, 10]",
        strategy=Eager(select_k=10, start_k=2),
    )

    assert await traversal(max_depth=0) == ["A0", "AR"]
    assert await traversal(max_depth=2) == ["A0", "AR", "B0", "BR", "T0", "TR"]


async def test_ids(sync_or_async: SyncOrAsync):
    embedding = angular_2d_embedding
    v0 = Content.new("v0", "-0.124", embedding)
    v1 = Content.new("v1", "+0.127", embedding, metadata={"mentions": ["v0"]})
    v2 = Content.new("v2", "+0.250", embedding, metadata={"mentions": ["v1", "v3"]})
    v3 = Content.new("v3", "+1.000", embedding)
    store = InMemory(embedding, [v0, v1, v2, v3])

    mentions_to_id = sync_or_async.traverse_sorted_ids(
        store=store,
        query="+0.249",
        strategy=Eager(start_k=1),
        edges=[("mentions", "$id")],
    )
    assert await mentions_to_id(max_depth=0) == ["v2"]
    assert await mentions_to_id(max_depth=1) == ["v1", "v2", "v3"]
    assert await mentions_to_id(max_depth=2) == ["v0", "v1", "v2", "v3"]

    id_to_mentions = sync_or_async.traverse_sorted_ids(
        store=store,
        query="-0.125",
        strategy=Eager(start_k=1),
        edges=[("$id", "mentions")],
    )
    assert await id_to_mentions(max_depth=0) == ["v0"]
    assert await id_to_mentions(max_depth=1) == ["v0", "v1"]
    assert await id_to_mentions(max_depth=2) == ["v0", "v1", "v2"]


async def test_edge_functions(sync_or_async: SyncOrAsync):
    embedding = angular_2d_embedding
    v0 = Content.new(
        "v0",
        "-0.124",
        embedding,
        metadata={"links": [("a", 5.0)], "incoming": ["a"]},
    )
    v1 = Content.new(
        "v1",
        "+1.000",
        embedding,
        metadata={"links": [("a", 6.0)], "incoming": ["a"]},
    )
    store = InMemory(embedding, [v0, v1])

    def link_function(node: Content) -> Edges:
        links = node.metadata.get("links", [])
        incoming = node.metadata.get("incoming", [])
        return Edges(
            incoming={MetadataEdge("incoming", v) for v in incoming},
            outgoing={MetadataEdge("incoming", v) for v, _weight in links},
        )

    traversal = sync_or_async.traverse_sorted_ids(
        store=store,
        query="-0.125",
        edges=link_function,
        strategy=Eager(start_k=1),
    )
    assert await traversal(max_depth=0) == ["v0"]
    assert await traversal(max_depth=1) == ["v0", "v1"]
