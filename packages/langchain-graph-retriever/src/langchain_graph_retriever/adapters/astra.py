from collections.abc import Sequence
from typing import (
    Any,
    override,
)

try:
    from langchain_astradb import AstraDBVectorStore
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-astradb`")
from langchain_core.documents import Document

from .base import METADATA_EMBEDDING_KEY, Adapter


class AstraAdapter(Adapter[AstraDBVectorStore]):
    """Adapter for AstraDBVectorSTore."""

    def __init__(self, vector_store: AstraDBVectorStore):
        super().__init__(vector_store, use_normalized_metadata=True)

    def _build_docs(
        self, docs_with_embeddings: list[tuple[Document, list[float]]]
    ) -> list[Document]:
        docs: list[Document] = []
        for doc, embedding in docs_with_embeddings:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            docs.append(doc)
        return docs

    @override
    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Document]]:
        query_embedding, docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return query_embedding, self._build_docs(
            docs_with_embeddings=docs_with_embeddings
        )

    @override
    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Document]]:
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self.vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, self._build_docs(
            docs_with_embeddings=docs_with_embeddings
        )

    @override
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_docs(docs_with_embeddings=docs_with_embeddings)

    @override
    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        docs_with_embeddings = (
            await self.vector_store.asimilarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_docs(docs_with_embeddings=docs_with_embeddings)

    @override
    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        docs: list[Document] = []
        for id in ids:
            doc = self._get_by_id_with_embedding(id)
            if doc is not None:
                docs.append(doc)
        return docs

    def _get_by_id_with_embedding(self, document_id: str) -> Document | None:
        self.vector_store.astra_env.ensure_db_setup()

        hit = self.vector_store.astra_env.collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        if hit is None:
            return None
        document = self.vector_store.document_codec.decode(hit)
        if document is None:
            return None
        document.metadata[METADATA_EMBEDDING_KEY] = (
            self.vector_store.document_codec.decode_vector(hit)
        )
        return document

    @override
    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        docs: list[Document] = []
        # TODO: Do this asynchronously?
        for id in ids:
            doc = await self._aget_by_id_with_embedding(id)
            if doc is not None:
                docs.append(doc)
        return docs

    async def _aget_by_id_with_embedding(self, document_id: str) -> Document | None:
        await self.vector_store.astra_env.aensure_db_setup()

        hit = await self.vector_store.astra_env.async_collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        if hit is None:
            return None
        document = self.vector_store.document_codec.decode(hit)
        if document is None:
            return None
        document.metadata[METADATA_EMBEDDING_KEY] = (
            self.vector_store.document_codec.decode_vector(hit)
        )
        return document
