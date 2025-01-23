from __future__ import annotations

from typing import Any, override

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_graph_retriever.adapters import Adapter, infer_adapter

from tests.integration_tests.stores import StoreFactory


def test_infer_store(store_factory: StoreFactory) -> None:
    # Some vector stores require at least one document to be created.
    doc = Document(
        id="doc",
        page_content="lorem ipsum and whatnot",
    )
    store = store_factory._create_store("foo", [doc], FakeEmbeddings(size=8))

    adapter = infer_adapter(store)

    assert isinstance(adapter, Adapter)
    if store_factory._teardown:
        store_factory._teardown(store)


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
