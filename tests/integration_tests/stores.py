import abc
import dataclasses
from typing import Callable, Generic, TypeVar

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from graph_pancake.retrievers.traversal_adapters.eager.traversal_adapter import (
    TraversalAdapter,
)
from graph_pancake.retrievers.traversal_adapters.generic import StoreAdapter
from graph_pancake.retrievers.traversal_adapters.mmr.mmr_traversal_adapter import (
    MMRTraversalAdapter,
)

ALL_STORES = ["mem", "mem_denorm", "astra", "cassandra", "chroma", "opensearch"]


@pytest.fixture(scope="session")
def enabled_stores(request: pytest.FixtureRequest) -> set[str]:
    # TODO: Use StrEnum?
    stores = request.config.getoption("--stores")

    if stores and "all" in stores:
        return set(ALL_STORES)
    elif stores:
        return set(stores)
    else:
        return {"mem", "mem_denorm"}


@pytest.fixture(scope="session", params=ALL_STORES)
def store_param(request: pytest.FixtureRequest, enabled_stores: set[str]) -> str:
    store: str = request.param
    if store not in enabled_stores:
        pytest.skip(f"'{store}' is not enabled")
    return store


T = TypeVar("T", bound=VectorStore)


# TODO: Eliminate when we only support generic.
@dataclasses.dataclass
class Stores:
    eager: TraversalAdapter
    mmr: MMRTraversalAdapter
    generic: StoreAdapter


class StoreFactory(abc.ABC, Generic[T]):
    def __init__(
        self,
        support_normalized_metadata: bool,
        create_store: Callable[[str, list[Document], Embeddings], T],
        create_eager: Callable[[T], TraversalAdapter],
        create_mmr: Callable[[T], MMRTraversalAdapter],
        create_generic: Callable[[T], StoreAdapter],
        teardown: Callable[[T], None] | None = None,
    ):
        self.support_normalized_metadata = support_normalized_metadata
        self._create_store = create_store
        self._create_eager = create_eager
        self._create_mmr = create_mmr
        self._create_generic = create_generic
        self._teardown = teardown
        self._index = 0

    def create(
        self,
        request: pytest.FixtureRequest,
        embedding: Embeddings,
        docs: list[Document],
    ) -> Stores:
        name = f"test_{self._index}"
        self._index += 1
        if not self.support_normalized_metadata:
            docs = list(MetadataDenormalizer().transform_documents(docs))
        store = self._create_store(name, docs, embedding)

        if self._teardown is not None:
            # make a local copy of the non-None teardown. This makes `mypy` happy.
            # Otherwise, it (correctly) recognizes that `self._teardown` could be not
            # `None` and `None` later (when the finalizer is called)
            teardown = self._teardown
            request.addfinalizer(lambda: teardown(store))

        return Stores(
            eager=self._create_eager(store),
            mmr=self._create_mmr(store),
            generic=self._create_generic(store),
        )


@pytest.fixture(scope="session")
def store_factory(store_param: str, request: pytest.FixtureRequest) -> StoreFactory:
    if store_param == "mem" or store_param == "mem_denorm":
        support_normalized_metadata = not store_param.endswith("_denorm")

        from langchain_core.vectorstores import InMemoryVectorStore

        from graph_pancake.retrievers.traversal_adapters.eager.in_memory_traversal_adapter import (  # noqa: E501
            InMemoryTraversalAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.generic.in_memory import (
            InMemoryStoreAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.mmr.in_memory_mmr_traversal_adapter import (  # noqa: E501
            InMemoryMMRTraversalAdapter,
        )

        return StoreFactory[InMemoryVectorStore](
            support_normalized_metadata=support_normalized_metadata,
            create_store=lambda _name, docs, emb: InMemoryVectorStore.from_documents(
                docs, emb
            ),
            create_eager=lambda store: InMemoryTraversalAdapter(
                store, support_normalized_metadata
            ),
            create_mmr=lambda store: InMemoryMMRTraversalAdapter(
                store, support_normalized_metadata
            ),
            create_generic=lambda store: InMemoryStoreAdapter(
                store, support_normalized_metadata=support_normalized_metadata
            ),
        )
    elif store_param == "chroma":
        from langchain_chroma.vectorstores import Chroma

        from graph_pancake.retrievers.traversal_adapters.eager.chroma_traversal_adapter import (  # noqa: E501
            ChromaTraversalAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.generic.chroma import (
            ChromaStoreAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.mmr.chroma_mmr_traversal_adapter import (  # noqa: E501
            ChromaMMRTraversalAdapter,
        )

        return StoreFactory[Chroma](
            support_normalized_metadata=False,
            create_store=lambda name, docs, emb: Chroma.from_documents(
                docs, emb, collection_name=name
            ),
            create_eager=ChromaTraversalAdapter,
            create_mmr=ChromaMMRTraversalAdapter,
            create_generic=ChromaStoreAdapter,
            teardown=lambda store: store.delete_collection(),
        )
    elif store_param == "astra":
        from langchain_astradb import AstraDBVectorStore

        from graph_pancake.retrievers.traversal_adapters.eager.astra_traversal_adapter import (  # noqa: E501
            AstraTraversalAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.generic.astra import (
            AstraStoreAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.mmr.astra_mmr_traversal_adapter import (  # noqa: E501
            AstraMMRTraversalAdapter,
        )

        def create_astra(
            name: str, docs: list[Document], embedding: Embeddings
        ) -> AstraDBVectorStore:
            try:
                import os

                from astrapy.authentication import StaticTokenProvider
                from dotenv import load_dotenv
                from langchain_astradb import AstraDBVectorStore

                load_dotenv()

                token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])

                store = AstraDBVectorStore(
                    embedding=embedding,
                    collection_name=name,
                    namespace="default_keyspace",
                    token=token,
                    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
                )
                store.add_documents(docs)
                return store

            except (ImportError, ModuleNotFoundError):
                msg = (
                    "to test graph-traversal with AstraDB, please"
                    " install langchain-astradb and python-dotenv"
                )
                raise ImportError(msg)

        def teardown_astra(store: AstraDBVectorStore):
            store.delete_collection()

        return StoreFactory[AstraDBVectorStore](
            support_normalized_metadata=True,
            create_store=create_astra,
            create_eager=AstraTraversalAdapter,
            create_mmr=AstraMMRTraversalAdapter,
            create_generic=AstraStoreAdapter,
            teardown=teardown_astra,
        )
    elif store_param == "cassandra":
        import os

        from cassandra.cluster import Cluster  # type: ignore
        from langchain_community.vectorstores.cassandra import Cassandra

        from graph_pancake.retrievers.traversal_adapters.eager.cassandra_traversal_adapter import (  # noqa: E501
            CassandraTraversalAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.generic.cassandra import (
            CassandraStoreAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.mmr.cassandra_mmr_traversal_adapter import (  # noqa: E501
            CassandraMMRTraversalAdapter,
        )

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

        KEYSPACE = "graph_test_keyspace"
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )

        request.addfinalizer(lambda: cluster.shutdown())

        def create_cassandra(
            name: str, docs: list[Document], embedding: Embeddings
        ) -> Cassandra:
            session = cluster.connect()
            session.execute(f"DROP TABLE IF EXISTS {KEYSPACE}.{name}")

            store = Cassandra(
                embedding=embedding,
                session=session,
                keyspace=KEYSPACE,
                table_name=name,
            )
            store.add_documents(docs)
            return store

        def teardown_cassandra(cassandra: Cassandra):
            assert cassandra.session is not None
            cassandra.session.shutdown()

        return StoreFactory[Cassandra](
            support_normalized_metadata=False,
            create_store=create_cassandra,
            create_eager=CassandraTraversalAdapter,
            create_mmr=CassandraMMRTraversalAdapter,
            create_generic=CassandraStoreAdapter,
            teardown=teardown_cassandra,
        )
    elif store_param == "opensearch":
        from langchain_community.vectorstores import OpenSearchVectorSearch

        from graph_pancake.retrievers.traversal_adapters.eager.open_search_traversal_adapter import (  # noqa: E501
            OpenSearchTraversalAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.generic.open_search import (
            OpenSearchStoreAdapter,
        )
        from graph_pancake.retrievers.traversal_adapters.mmr.open_search_mmr_traversal_adapter import (  # noqa: E501
            OpenSearchMMRTraversalAdapter,
        )

        def create_open_search(
            name: str, docs: list[Document], embedding: Embeddings
        ) -> OpenSearchVectorSearch:
            store = OpenSearchVectorSearch(
                opensearch_url="http://localhost:9200",
                index_name=name,
                embedding_function=embedding,
                engine="faiss",
            )
            store.add_documents(docs)
            return store

        def teardown_open_search(store: OpenSearchVectorSearch) -> None:
            if store.index_exists():
                store.delete_index()

        return StoreFactory[OpenSearchVectorSearch](
            support_normalized_metadata=False,
            create_store=create_open_search,
            create_eager=OpenSearchTraversalAdapter,
            create_mmr=OpenSearchMMRTraversalAdapter,
            create_generic=OpenSearchStoreAdapter,
            teardown=teardown_open_search,
        )
    else:
        pytest.fail(f"Unsupported store: {store_param}")


@pytest.fixture(scope="session")
def support_normalized_metadata(store_factory: StoreFactory) -> bool:
    return store_factory.support_normalized_metadata
