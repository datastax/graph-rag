from collections.abc import Iterator

import pytest
from graph_retriever import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)


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

        metadata_denormalizer = MetadataDenormalizer()
        docs = list(metadata_denormalizer.transform_documents(animal_docs))
        store = Chroma.from_documents(
            docs, animal_embeddings, collection_name="animals"
        )

        yield ChromaAdapter(
            store, metadata_denormalizer, nested_metadata_fields={"keywords"}
        )

        store.delete_collection()
