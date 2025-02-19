import typing
from collections.abc import Iterator
from typing import Union

import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.transformers import ShreddingTransformer

if typing.TYPE_CHECKING:
    from cassandra.cluster import Cluster  # type: ignore


@pytest.fixture(scope="module")
def cluster(
    request: pytest.FixtureRequest, enabled_stores: set[str], testcontainers: set[str]
) -> Iterator[Union["Cluster", None]]:
    if "cassandra" not in enabled_stores:
        pytest.skip("Pass --stores=cassandra to test Cassandra")
        return

    import os

    from cassandra.cluster import Cluster  # type: ignore

    if "cassandra" in testcontainers:
        from testcontainers.cassandra import CassandraContainer  # type: ignore

        container = CassandraContainer(image="cassandra:5.0.2")
        container.start()

        request.addfinalizer(lambda: container.stop())
        contact_points = container.get_contact_points()
    elif "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None

    cluster = Cluster(contact_points)
    yield cluster
    cluster.shutdown()


class TestCassandraAdapter(AdapterComplianceSuite):
    def supports_nested_metadata(self) -> bool:
        return False

    @pytest.fixture(scope="class")
    def adapter(
        self,
        cluster: "Cluster",
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator[Adapter]:
        from langchain_community.vectorstores.cassandra import Cassandra
        from langchain_graph_retriever.adapters.cassandra import (
            CassandraAdapter,
        )

        session = cluster.connect()

        KEYSPACE = "graph_test_keyspace"
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}"
            " WITH replication = "
            "{'class': 'SimpleStrategy', 'replication_factor': 1}"
        )

        shredder = ShreddingTransformer()
        session = cluster.connect()
        session.execute(f"DROP TABLE IF EXISTS {KEYSPACE}.animals")
        store = Cassandra(
            embedding=animal_embeddings,
            session=session,
            keyspace=KEYSPACE,
            table_name="animals",
        )
        docs = list(shredder.transform_documents(animal_docs))
        store.add_documents(docs)
        yield CassandraAdapter(store, shredder, {"keywords", "tags"})

        if session:
            session.shutdown()
