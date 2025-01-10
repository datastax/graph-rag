"""Test of Apache Cassandra graph index class `Cassandra`"""

import os
from contextlib import contextmanager
from typing import Any, Generator, Iterable

import pytest
from langchain_chroma import Chroma
from langchain_community.vectorstores import Cassandra, OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pytest import FixtureRequest

from graph_pancake.retrievers.graph_mmr_traversal_retriever import (
    GraphMMRTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.mmr import (
    AstraMMRTraversalAdapter,
    CassandraMMRTraversalAdapter,
    ChromaMMRTraversalAdapter,
    MMRTraversalAdapter,
    OpenSearchMMRTraversalAdapter,
)
from tests.embeddings import (
    AngularTwoDimensionalEmbeddings,
    ParserEmbeddings,
)

vector_store_types = [
    "astra-db",
    "cassandra",
    "chroma-db",
    "open-search",
]


def _doc_ids(docs: Iterable[Document]) -> set[str]:
    return {doc.id for doc in docs if doc.id is not None}


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
    from cassandra.cluster import Cluster

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


@pytest.fixture(scope="function")
def vector_store(
    request: FixtureRequest, embedding_type: str, vector_store_type: str
) -> Generator[VectorStore, None, None]:
    embeddings: Embeddings
    if embedding_type == "angular-embeddings":
        embeddings = AngularTwoDimensionalEmbeddings()
    elif embedding_type == "d2-embeddings":
        embeddings = ParserEmbeddings(dimension=2)
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
                "to test mmr-graph-traversal with AstraDB, please"
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
        if store.index_exists:
            store.delete_index()  # store.index_name
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


def get_adapter(
    vector_store: VectorStore, vector_store_type: str
) -> MMRTraversalAdapter:
    if vector_store_type == "astra-db":
        return AstraMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "cassandra":
        return CassandraMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "chroma-db":
        return ChromaMMRTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "open-search":
        return OpenSearchMMRTraversalAdapter(vector_store=vector_store)
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["angular-embeddings"])
def test_mmr_traversal(vector_store: VectorStore, vector_store_type: str) -> None:
    """Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

            ______ v2
          //      \\
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

    vector_store.add_documents([v0, v1, v2, v3])

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphMMRTraversalRetriever(
        store=vector_store_adapter,
        edges=[("outgoing", "incoming")],
        fetch_k=2,
        k=2,
        depth=2,
    )

    docs = retriever.invoke("0.0", k=2, fetch_k=2)
    assert _doc_ids(docs) == {"v0", "v2"}

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = retriever.invoke("0.0", k=2, fetch_k=2, depth=0)
    assert _doc_ids(docs) == {"v0", "v1"}

    # With max depth 0 but higher `fetch_k`, we encounter v2
    docs = retriever.invoke("0.0", k=2, fetch_k=3, depth=0)
    assert _doc_ids(docs) == {"v0", "v2"}

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    assert _doc_ids(docs) == {"v0"}

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert _doc_ids(docs) == {"v0", "v2", "v1", "v3"}


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


class TestMmrGraphTraversal:
    @pytest.mark.parametrize("vector_store_type", vector_store_types)
    @pytest.mark.parametrize("embedding_type", ["d2-embeddings"])
    def test_invoke_sync(
        self,
        vector_store: VectorStore,
        vector_store_type: str,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a vector store."""
        vector_store.add_documents(graph_vector_store_docs)

        vector_store_adapter = get_adapter(
            vector_store=vector_store,
            vector_store_type=vector_store_type,
        )

        retriever = GraphMMRTraversalRetriever(
            store=vector_store_adapter,
            vector_store=vector_store,
            edges=[("out", "in"), "tag"],
            depth=2,
            k=2,
        )

        docs = retriever.invoke(input="[2, 10]")
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    @pytest.mark.parametrize("vector_store_type", vector_store_types)
    @pytest.mark.parametrize("embedding_type", ["d2-embeddings"])
    async def test_invoke_async(
        self,
        vector_store: VectorStore,
        vector_store_type: str,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a vector store."""
        await vector_store.aadd_documents(graph_vector_store_docs)

        vector_store_adapter = get_adapter(
            vector_store=vector_store,
            vector_store_type=vector_store_type,
        )

        retriever = GraphMMRTraversalRetriever(
            store=vector_store_adapter,
            vector_store=vector_store,
            edges=[("out", "in"), "tag"],
            depth=2,
            k=2,
        )
        mt_labels = set()
        docs = await retriever.ainvoke(input="[2, 10]")
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])
