import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore


class TestInMemory(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self,
        enabled_stores: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Adapter:
        if "mem" not in enabled_stores:
            pytest.skip("Pass --stores=mem to test InMemory")

        from langchain_graph_retriever.adapters.in_memory import (
            InMemoryAdapter,
        )

        store = InMemoryVectorStore.from_documents(animal_docs, animal_embeddings)
        return InMemoryAdapter(store)
