from __future__ import annotations

from typing import Any

import langchain_astradb
import pytest
from graph_retriever.adapters import Adapter
from langchain_chroma import Chroma
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_community.vectorstores.opensearch_vector_search import (
    OpenSearchVectorSearch,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore, VectorStore
from langchain_graph_retriever.adapters.astra import AstraAdapter
from langchain_graph_retriever.adapters.cassandra import CassandraAdapter
from langchain_graph_retriever.adapters.chroma import ChromaAdapter
from langchain_graph_retriever.adapters.inference import (
    _infer_adapter_name,
    infer_adapter,
)
from langchain_graph_retriever.adapters.open_search import OpenSearchAdapter
from typing_extensions import override


def test_infer_in_memory():
    store = InMemoryVectorStore(FakeEmbeddings(size=4))
    adapter = infer_adapter(store)
    assert isinstance(adapter, Adapter)


@pytest.mark.parametrize(
    "cls,adapter_cls",
    [
        (langchain_astradb.AstraDBVectorStore, AstraAdapter),
        (Cassandra, CassandraAdapter),
        (Chroma, ChromaAdapter),
        (OpenSearchVectorSearch, OpenSearchAdapter),
    ],
)
def test_infer_adapter_name(cls: type, adapter_cls: type) -> None:
    module_name, class_name = _infer_adapter_name(cls)
    assert module_name == adapter_cls.__module__
    assert class_name == adapter_cls.__name__


class UnsupportedVectorStore(VectorStore):
    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[Any, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> UnsupportedVectorStore:
        return UnsupportedVectorStore()

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return []


def test_infer_store_unknown():
    EXPECTED = (
        "Expected adapter or supported vector store, but got"
        f" {__name__}.UnsupportedVectorStore"
    )
    with pytest.raises(ValueError, match=EXPECTED):
        infer_adapter(UnsupportedVectorStore())
