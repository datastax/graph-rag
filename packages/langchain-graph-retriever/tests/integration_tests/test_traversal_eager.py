import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_graph_retriever import (
    GraphRetriever,
)
from langchain_graph_retriever.adapters.in_memory import InMemoryAdapter
from langchain_graph_retriever.edges.metadata import Id
from langchain_graph_retriever.strategies import (
    Eager,
)
from langchain_graph_retriever.types import Edges, MetadataEdge, Node

from tests.animal_docs import (
    ANIMALS_DEPTH_0_EXPECTED,
    ANIMALS_QUERY,
)
from tests.embeddings.simple_embeddings import (
    Angular2DEmbeddings,
    EarthEmbeddings,
    ParserEmbeddings,
)
from tests.integration_tests.assertions import assert_document_format, sorted_doc_ids
from tests.integration_tests.stores import Adapter, AdapterFactory


async def test_animals_bidir_collection_eager(animal_store: Adapter, invoker):
    # test graph-search on a normalized bi-directional edge
    retriever = GraphRetriever(
        store=animal_store,
        edges=[("keywords", "keywords")],
        strategy=Eager(k=100, start_k=2, max_depth=0),
    )

    docs: list[Document] = await invoker(retriever, ANIMALS_QUERY, max_depth=0)
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(retriever, ANIMALS_QUERY, max_depth=1)
    assert sorted_doc_ids(docs) == [
        "cat",
        "coyote",
        "fox",
        "gazelle",
        "hyena",
        "jackal",
        "mongoose",
    ]

    docs = await invoker(retriever, ANIMALS_QUERY, max_depth=2)
    assert sorted_doc_ids(docs) == [
        "alpaca",
        "bison",
        "cat",
        "chicken",
        "cockroach",
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


async def test_animals_bidir_item(animal_store: Adapter, invoker):
    retriever = GraphRetriever(
        store=animal_store,
        edges=[("habitat", "habitat")],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == [
        "bobcat",
        "cobra",
        "deer",
        "elk",
        "fox",
        "mongoose",
    ]


async def test_animals_initial_roots(animal_store: Adapter, invoker):
    retriever = GraphRetriever(
        store=animal_store,
        edges=[("keywords", "keywords")],
        strategy=Eager(k=10, start_k=0),
    )

    docs = await invoker(
        retriever,
        ANIMALS_QUERY,
        initial_roots=["bobcat"],
        max_depth=0,
    )

    # bobcat is included (initial roots).
    # everything adjacent to bobcat is depth 0 (immediately reachable)
    assert sorted_doc_ids(docs) == [
        "bear",
        "bobcat",
    ]

    docs = await invoker(
        retriever,
        ANIMALS_QUERY,
        initial_roots=["bobcat"],
        max_depth=1,
    )

    assert sorted_doc_ids(docs) == [
        "bear",
        "bobcat",
        "moose",
        "ostrich",
    ]

    docs = await invoker(
        retriever,
        ANIMALS_QUERY,
        initial_roots=["bobcat", "cheetah"],
        k=20,
        max_depth=1,
    )

    assert sorted_doc_ids(docs) == [
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
        "ostrich",
    ]


async def test_animals_item_to_collection(animal_store: Adapter, invoker):
    retriever = GraphRetriever(
        store=animal_store,
        edges=[("habitat", "keywords")],
    )

    docs: list[Document] = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ANIMALS_DEPTH_0_EXPECTED

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]

    docs = await invoker(
        retriever, ANIMALS_QUERY, strategy=Eager(k=10, start_k=2, max_depth=2)
    )
    assert sorted_doc_ids(docs) == ["bear", "bobcat", "caribou", "fox", "mongoose"]


async def test_parser(
    request: pytest.FixtureRequest, adapter_factory: AdapterFactory, invoker
):
    """
    This is a test of set of Documents to pre-populate,
    a graph vector store with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
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

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BR", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
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

    retriever = GraphRetriever(
        store=adapter_factory.create(request, ParserEmbeddings(dimension=2), documents),
        edges=[("out", "in"), ("tag", "tag")],
        strategy=Eager(k=10, start_k=2, max_depth=2),
    )

    docs: list[Document] = await invoker(
        retriever, "[2, 10]", strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    ss_labels = {doc.metadata["label"] for doc in docs}
    assert ss_labels == {"AR", "A0"}
    assert_document_format(docs[0])

    docs = await invoker(retriever, "[2, 10]")
    # this is a set, as some of the internals of trav.search are set-driven
    # so ordering is not deterministic:
    ts_labels = {doc.metadata["label"] for doc in docs}
    assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
    assert_document_format(docs[0])


async def test_earth(
    request: pytest.FixtureRequest, adapter_factory: AdapterFactory, invoker
):
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )

    retriever = GraphRetriever(
        store=adapter_factory.create(
            request,
            EarthEmbeddings(),
            [greetings, doc1, doc2],
            nested_metadata_fields={"keywords"},
        ),
        edges=[("outgoing", "incoming"), ("keywords", "keywords")],
        strategy=Eager(k=10, start_k=2, max_depth=0),
    )

    docs: list[Document] = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=1, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=2, max_depth=0)
    )
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = await invoker(
        retriever, "Earth", strategy=Eager(k=10, start_k=1, max_depth=1)
    )
    assert sorted_doc_ids(docs) == ["doc1", "doc2", "greetings"]


async def test_mentions_to_ids(invoker) -> None:
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127", metadata={"mentions": ["v0"]})
    v2 = Document(id="v2", page_content="+0.250", metadata={"mentions": ["v1", "v3"]})
    v3 = Document(id="v3", page_content="+1.000")

    store = InMemoryVectorStore(embedding=Angular2DEmbeddings())
    store.add_documents([v0, v1, v2, v3])

    retriever = GraphRetriever(
        store=InMemoryAdapter(vector_store=store),
        edges=[("mentions", Id())],
    )

    docs: list[Document] = await invoker(retriever, "+0.249", start_k=1, max_depth=0)
    assert sorted_doc_ids(docs) == ["v2"]

    docs = await invoker(retriever, "+0.249", start_k=1, max_depth=1)
    assert sorted_doc_ids(docs) == ["v1", "v2", "v3"]

    docs = await invoker(retriever, "+0.249", start_k=1, max_depth=3)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]


async def test_ids_to_mentions(invoker) -> None:
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127", metadata={"mentions": ["v0"]})
    v2 = Document(id="v2", page_content="+0.250", metadata={"mentions": ["v1", "v3"]})
    v3 = Document(id="v3", page_content="+1.000")

    store = InMemoryVectorStore(embedding=Angular2DEmbeddings())
    store.add_documents([v0, v1, v2, v3])

    retriever = GraphRetriever(
        store=InMemoryAdapter(vector_store=store),
        edges=[(Id(), "mentions")],
    )

    docs: list[Document] = await invoker(retriever, "-0.125", start_k=1, max_depth=0)
    assert sorted_doc_ids(docs) == ["v0"]

    docs = await invoker(retriever, "-0.125", start_k=1, max_depth=1)
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    docs = await invoker(retriever, "-0.125", start_k=1, max_depth=3)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2"]


async def test_edge_function(invoker) -> None:
    v0 = Document(
        id="v0",
        page_content="-0.124",
        metadata={"links": [("a", 5.0)], "incoming": ["a"]},
    )
    v1 = Document(
        id="v1",
        page_content="+1.000",
        metadata={"links": [("a", 6.0)], "incoming": ["a"]},
    )

    store = InMemoryVectorStore(embedding=Angular2DEmbeddings())
    store.add_documents([v0, v1])

    def link_function(node: Node) -> Edges:
        links = node.metadata.get("links", [])
        incoming = node.metadata.get("incoming", [])
        return Edges(
            incoming={MetadataEdge("incoming", v) for v in incoming},
            outgoing={MetadataEdge("incoming", v) for v, _weight in links},
        )

    retriever = GraphRetriever(
        store=InMemoryAdapter(vector_store=store),
        edges=link_function,
    )

    docs: list[Document] = await invoker(retriever, "-0.125", start_k=1, max_depth=0)
    assert sorted_doc_ids(docs) == ["v0"]

    docs = await invoker(retriever, "-0.125", start_k=1, max_depth=1)
    assert sorted_doc_ids(docs) == ["v0", "v1"]
