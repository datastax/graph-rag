"""Provides an adapter for the InMemoryVectorStore integration."""

from collections.abc import Sequence
from typing import override

from langchain_core.documents import Document

from ..vector_stores.in_memory import InMemoryFlat, InMemoryList
from .base import METADATA_EMBEDDING_KEY, Adapter, DenormalizedAdapter

SENTINEL = object()


class InMemoryFlatAdapter(DenormalizedAdapter[InMemoryFlat]):
    """
    Adapter for InMemoryFlat vector store.

    This adapter integrates the in-memory-flat vector store with the graph
    retriever system, enabling similarity search and document retrieval.

    Parameters
    ----------
    vector_store : InMemoryFlat
        The in-memory-flat vector store instance.
    metadata_denormalizer: MetadataDenormalizer | None
        (Optional) An instance of the MetadataDenormalizer used for doc insertion.
        If not passed then a default instance of MetadataDenormalizer is used.
    """

    @override
    def get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        docs: list[Document] = []

        for doc_id in ids:
            doc = self.vector_store.store.get(doc_id)
            if doc:
                metadata = doc["metadata"]
                metadata[METADATA_EMBEDDING_KEY] = doc["vector"]
                docs.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=metadata,
                    )
                )
        return list(self.metadata_denormalizer.revert_documents(docs))

    @override
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs,
    ):
        results = self.vector_store._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        docs = [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={METADATA_EMBEDDING_KEY: doc_embedding, **doc.metadata},
            )
            for doc, _score, doc_embedding in results
        ]
        return list(self.metadata_denormalizer.revert_documents(docs))


class InMemoryListAdapter(Adapter[InMemoryList]):
    """
    Adapter for InMemoryList vector store.

    This adapter integrates the in-memory-list vector store with the graph
    retriever system, enabling similarity search and document retrieval.

    Parameters
    ----------
    vector_store : InMemoryList
        The in-memory-list vector store instance.
    """

    @override
    def get(self, ids: Sequence[str], /, **kwargs) -> list[Document]:
        docs: list[Document] = []

        for doc_id in ids:
            doc = self.vector_store.store.get(doc_id)
            if doc:
                metadata = doc["metadata"]
                metadata[METADATA_EMBEDDING_KEY] = doc["vector"]
                docs.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=metadata,
                    )
                )
        return docs

    @override
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs,
    ):
        results = self.vector_store._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        docs = [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={METADATA_EMBEDDING_KEY: doc_embedding, **doc.metadata},
            )
            for doc, _score, doc_embedding in results
        ]
        return docs
