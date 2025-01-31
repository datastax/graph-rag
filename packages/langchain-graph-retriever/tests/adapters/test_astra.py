from collections.abc import Iterator

import pytest
from graph_retriever import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TestAstraAdapter(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self,
        enabled_stores: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator[Adapter]:
        if "astra" not in enabled_stores:
            pytest.skip("Pass --stores=astra to test Astra")
            return

        import os

        from astrapy import AstraDBDatabaseAdmin
        from astrapy.authentication import StaticTokenProvider
        from dotenv import load_dotenv
        from langchain_astradb import AstraDBVectorStore
        from langchain_graph_retriever.adapters.astra import (
            AstraAdapter,
        )

        load_dotenv()

        token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
        keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
        api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

        admin = AstraDBDatabaseAdmin(api_endpoint=api_endpoint, token=token)
        admin.create_keyspace(keyspace)

        store = AstraDBVectorStore(
            embedding=animal_embeddings,
            collection_name="animals",
            namespace=keyspace,
            token=token,
            api_endpoint=api_endpoint,
            pre_delete_collection=True,
        )
        store.add_documents(animal_docs)

        yield AstraAdapter(store)

        store.delete_collection()
