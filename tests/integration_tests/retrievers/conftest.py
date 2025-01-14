import json
import os
from contextlib import contextmanager
from typing import Any, Generator, Iterable, cast

import pytest
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import Cassandra, OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from pytest import Metafunc

from graph_pancake.retrievers.traversal_adapters.generic.astra import (
    AstraStoreAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic.base import (
    StoreAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic.cassandra import (
    CassandraStoreAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic.chroma import (
    ChromaStoreAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic.in_memory import (
    InMemoryStoreAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic.open_search import (
    OpenSearchStoreAdapter,
)
from tests.embeddings import (
    AngularTwoDimensionalEmbeddings,
    EarthEmbeddings,
    ParserEmbeddings,
)
from tests.embeddings.simple_embeddings import AnimalEmbeddings


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Dynamically generate parameters for pytest tests."""
    if "vector_store_type" in metafunc.fixturenames:
        # Use getfixturevalue to safely resolve the `vector_store_types` fixture
        in_memory_only = metafunc.config.getoption("--in-memory-only")

        if in_memory_only:
            vector_store_values = ["in-memory", "in-memory-denormalized"]
        else:
            vector_store_values = [
                "cassandra",
                #"chroma-db",
                "in-memory",
                #"in-memory-denormalized",
                #"open-search",
            ]

        # Parametrize the `vector_store_type` dynamically
        metafunc.parametrize("vector_store_type", vector_store_values)


def sorted_doc_ids(docs: Iterable[Document]) -> list[str]:
    return sorted([doc.id for doc in docs if doc.id is not None])


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


def supports_normalized_metadata(vector_store_type: str) -> bool:
    if vector_store_type in ["astra-db", "in-memory", "open-search"]:
        return True
    elif vector_store_type in [
        "cassandra",
        "chroma-db",
        "in-memory-denormalized",
    ]:
        return False
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


@pytest.fixture(scope="module")
def animal_docs() -> list[Document]:
    documents = []
    with open("tests/data/animals.jsonl", "r") as file:
        for line in file:
            data = json.loads(line.strip())
            documents.append(
                Document(
                    id=data["id"],
                    page_content=data["text"],
                    metadata=data["metadata"],
                )
            )

    return documents


@pytest.fixture(scope="module")
def hello_docs() -> list[Document]:
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
    return [greetings, doc1, doc2]


@pytest.fixture(scope="module")
def mmr_docs() -> list[Document]:
    """The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

            ______ v2
          //     \\
         //        \\  v1
    v3   |     .    | query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    return [v0, v1, v2, v3]


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

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
    return docs_a + docs_b + docs_f + docs_t


@pytest.fixture(scope="module")
def animal_store(animal_docs: list[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=AnimalEmbeddings())
    store.add_documents(animal_docs)
    return store


class CassandraSession:
    keyspace: str
    table_name: str
    session: Any

    def __init__(self, keyspace: str, table_name: str, session: Any):
        self.keyspace = keyspace
        self.table_name = table_name
        self.session = session


@contextmanager
def get_cassandra_session(
    keyspace: str, table_name: str, drop: bool = True
) -> Generator[CassandraSession, None, None]:
    """Initialize the Cassandra cluster and session"""
    from cassandra.cluster import Cluster  # type: ignore

    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None

    cluster = Cluster(contact_points)
    session = cluster.connect()

    try:
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {keyspace}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(
            keyspace=keyspace, table_name=table_name, session=session
        )
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@pytest.fixture
def vector_store(
    embedding_type: str, vector_store_type: str
) -> Generator[VectorStore, None, None]:
    embeddings: Embeddings
    if embedding_type == "earth":
        embeddings = EarthEmbeddings()
    elif embedding_type == "parser-d2":
        embeddings = ParserEmbeddings(dimension=2)
    elif embedding_type == "angular":
        embeddings = AngularTwoDimensionalEmbeddings()
    elif embedding_type == "animal":
        embeddings = AnimalEmbeddings()
    else:
        msg = f"Unknown embeddings type: {embedding_type}"
        raise ValueError(msg)

    store: VectorStore
    if vector_store_type == "astra-db":
        try:
            from astrapy.authentication import StaticTokenProvider
            from dotenv import load_dotenv
            from langchain_astradb import AstraDBVectorStore

            load_dotenv()

            token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])

            store = AstraDBVectorStore(
                embedding=embeddings,
                collection_name="graph_test_collection",
                namespace="default_keyspace",
                token=token,
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            )
            yield store
            store.delete_collection()

        except (ImportError, ModuleNotFoundError):
            msg = (
                "to test graph-traversal with AstraDB, please"
                " install langchain-astradb and python-dotenv"
            )
            raise ImportError(msg)

    elif vector_store_type == "cassandra":
        with get_cassandra_session(
            table_name="graph_test_table", keyspace="graph_test_keyspace"
        ) as session:
            store = Cassandra(
                embedding=embeddings,
                session=session.session,
                keyspace=session.keyspace,
                table_name=session.table_name,
            )
            yield store
    elif vector_store_type == "chroma-db":
        store = Chroma(embedding_function=embeddings)
        yield store
        store.delete_collection()
    elif vector_store_type == "open-search":
        store = OpenSearchVectorSearch(
            opensearch_url="http://localhost:9200",
            index_name="graph_test_index",
            embedding_function=embeddings,
            engine="faiss",
        )
        yield store
        if store.index_exists():
            store.delete_index()  # store.index_name
    elif vector_store_type in ["in-memory", "in-memory-denormalized"]:
        store = InMemoryVectorStore(embedding=embeddings)
        yield store
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


@pytest.fixture
def store_adapter(vector_store: VectorStore, vector_store_type: str) -> StoreAdapter:
    if vector_store_type == "astra-db":
        return AstraStoreAdapter(vector_store=cast(AstraDBVectorStore, vector_store))
    elif vector_store_type == "cassandra":
        return CassandraStoreAdapter(vector_store=cast(Cassandra, vector_store))
    elif vector_store_type == "chroma-db":
        return ChromaStoreAdapter(vector_store=cast(Chroma, vector_store))
    elif vector_store_type == "open-search":
        return OpenSearchStoreAdapter(
            vector_store=cast(OpenSearchVectorSearch, vector_store)
        )
    elif vector_store_type == "in-memory":
        return InMemoryStoreAdapter(
            vector_store=cast(InMemoryVectorStore, vector_store),
            support_normalized_metadata=True,
        )
    elif vector_store_type == "in-memory-denormalized":
        return InMemoryStoreAdapter(
            vector_store=cast(InMemoryVectorStore, vector_store),
            support_normalized_metadata=False,
        )
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)
