from collections.abc import Iterator

import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.transformers import ShreddingTransformer


class TestChroma(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self,
        enabled_stores: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator[Adapter]:
        if "chroma" not in enabled_stores:
            pytest.skip("Pass --stores=chroma to test Chroma")

        from langchain_chroma.vectorstores import Chroma
        from langchain_graph_retriever.adapters.chroma import (
            ChromaAdapter,
        )

        shredder = ShreddingTransformer()
        docs = list(shredder.transform_documents(animal_docs))
        store = Chroma.from_documents(
            docs,
            animal_embeddings,
            collection_name="animals",
            # Use `cosine` metric for consistency with other systems.
            # Default was L2.
            collection_metadata={"hnsw:space": "cosine"},
        )

        yield ChromaAdapter(store, shredder, nested_metadata_fields={"keywords"})

        store.delete_collection()
