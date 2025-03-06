from typing import cast

import networkx as nx
import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.document_graph import create_graph, group_by_community


@pytest.fixture
def docs() -> list[Document]:
    doc0 = Document(
        id="doc0",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={
            "outgoing": "parent",
            "keywords": ["greeting", "world"],
            "mentions": ["doc0"],
        },
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={
            "outgoing": "parent",
            "keywords": ["greeting", "earth"],
            "mentions": ["doc1", "doc0"],
        },
    )
    return [doc0, doc1, doc2]


def test_create_graph(docs: list[Document]):
    graph = create_graph(
        docs,
        edges=[("outgoing", "incoming")],
    )
    assert sorted(nx.to_edgelist(graph)) == [
        ("doc1", "doc0", {}),
        ("doc2", "doc0", {}),
    ]

    graph = create_graph(
        docs,
        edges=[("keywords", "keywords")],
    )
    assert sorted(nx.to_edgelist(graph)) == [
        ("doc1", "doc2", {}),
        ("doc2", "doc1", {}),
    ]

    graph = create_graph(
        docs,
        edges=[("outgoing", "incoming"), ("keywords", "keywords")],
    )
    assert sorted(nx.to_edgelist(graph)) == [
        ("doc1", "doc0", {}),
        ("doc1", "doc2", {}),
        ("doc2", "doc0", {}),
        ("doc2", "doc1", {}),
    ]
    assert graph.nodes["doc1"]["doc"] == docs[1]

    graph = create_graph(
        docs,
        edges=[("mentions", "$id")],
    )
    assert sorted(nx.to_edgelist(graph)) == [
        ("doc1", "doc0", {}),
        ("doc2", "doc0", {}),
        ("doc2", "doc1", {}),
    ]


def test_communities(animal_docs: list[Document]):
    graph = create_graph(
        animal_docs,
        edges=[("habitat", "habitat")],
    )
    communities = group_by_community(graph)

    community_ids = [sorted([cast(str, d.id) for d in c]) for c in communities]
    assert community_ids == [
        ["aardvark"],
        ["albatross", "barracuda", "crab"],
        ["alligator"],
        ["alpaca"],
        ["ant"],
        ["anteater"],
        ["antelope", "buffalo", "coyote", "hedgehog"],
        ["armadillo"],
        ["baboon"],
        ["badger"],
        ["bat"],
        ["bear"],
        ["beaver"],
        ["bee"],
        ["beetle"],
        ["bison"],
        ["blue jay"],
        ["boar"],
        ["bobcat", "cobra", "deer", "elk", "mongoose"],
        ["butterfly"],
        ["camel"],
        [
            "capybara",
            "crane",
            "crocodile",
            "dragonfly",
            "duck",
            "frog",
            "heron",
            "newt",
        ],
        ["caribou"],
        ["cassowary", "jaguar"],
        ["cat"],
        ["caterpillar"],
        ["chameleon", "chimpanzee", "gecko", "gorilla", "iguana", "leopard"],
        ["cheetah", "gazelle", "hyena", "lion", "ostrich"],
        ["chicken"],
        ["chinchilla"],
        ["cockroach"],
        ["crow"],
        ["dingo"],
        ["dog"],
        ["dolphin", "jellyfish"],
        ["donkey"],
        ["dove"],
        ["eagle"],
        ["eel"],
        ["elephant"],
        ["emu"],
        ["falcon"],
        ["ferret"],
        ["finch"],
        ["fish"],
        ["flamingo"],
        ["fox"],
        ["giraffe"],
        ["goat"],
        ["goose"],
        ["grasshopper"],
        ["guinea pig"],
        ["hamster"],
        ["hawk"],
        ["hippopotamus"],
        ["hornet"],
        ["horse"],
        ["hummingbird"],
        ["jackal"],
        ["kangaroo"],
        ["koala"],
        ["komodo dragon"],
        ["lark"],
        ["lemur"],
        ["lizard"],
        ["lobster"],
        ["magpie"],
        ["manatee"],
        ["moose"],
        ["mosquito"],
        ["narwhal"],
        ["octopus"],
    ]


def test_communities_disconnected_graph():
    graph = nx.DiGraph()
    graph.add_node("n1", doc=Document(id="n1", page_content="n1"))
    graph.add_node("n2", doc=Document(id="n2", page_content="n2"))

    communities = group_by_community(graph)

    community_ids = [sorted([cast(str, d.id) for d in c]) for c in communities]
    assert community_ids == [["n1"], ["n2"]]
