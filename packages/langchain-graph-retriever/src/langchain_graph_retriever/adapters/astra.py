"""Provides an adapter for AstraDB vector store integration."""

from collections.abc import Sequence
from typing import Any

import backoff
from graph_retriever.content import Content
from typing_extensions import override

try:
    from langchain_astradb import AstraDBVectorStore
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-astradb`")

try:
    import astrapy
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install astrapy")
import httpx
from graph_retriever.adapters import Adapter
from langchain_core.documents import Document

_EXCEPTIONS_TO_RETRY = (
    httpx.TransportError,
    astrapy.exceptions.DataAPIException,
)
_MAX_RETRIES = 3


class AstraAdapter(Adapter):
    """
    Adapter for AstraDBVectorStore.

    This adapter provides DataStax AstraDB support for the graph retriever
    system, enabling similarity search and document retrieval.

    It supports normalized metadata (collections of values) without
    denormalization.

    Parameters
    ----------
    vector_store : AstraDBVectorStore
        The AstraDB vector store instance.
    """

    def __init__(self, vector_store: AstraDBVectorStore) -> None:
        self.vector_store = vector_store

    @override
    def embed_query(self, query: str) -> list[float]:
        embedding = self.vector_store.embedding
        assert embedding is not None

        return embedding.embed_query(query)

    def _build_contents(
        self, docs_with_embeddings: list[tuple[Document, list[float]]]
    ) -> list[Content]:
        contents = []
        for doc, embedding in docs_with_embeddings:
            assert doc.id is not None
            contents.append(
                Content(
                    id=doc.id,
                    content=doc.page_content,
                    metadata=doc.metadata,
                    embedding=embedding,
                )
            )
        return contents

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding, docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return query_embedding, self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self.vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs_with_embeddings = (
            await self.vector_store.asimilarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_contents(docs_with_embeddings)

    @override
    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Content]:
        contents: list[Content] = []
        for id in set(ids):
            content = self._get_by_id_with_embedding(id)
            if content is not None:
                contents.append(content)
        return contents

    def _hit_to_content(self, hit: dict[str, Any] | None) -> Content | None:
        if hit is None:
            return None
        doc = self.vector_store.document_codec.decode(hit)
        if doc is None:
            return None
        assert doc.id is not None
        embedding = self.vector_store.document_codec.decode_vector(hit)
        assert embedding is not None
        return Content(
            id=doc.id,
            content=doc.page_content,
            metadata=doc.metadata,
            embedding=embedding,
        )

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def _get_by_id_with_embedding(self, document_id: str) -> Content | None:
        """
        Retrieve a document by its ID, including its embedding.

        Parameters
        ----------
        document_id : str
            The document ID.

        Returns
        -------
        Content | None
            The retrieved document with embedding, or `None` if not found.
        """
        self.vector_store.astra_env.ensure_db_setup()

        hit = self.vector_store.astra_env.collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        return self._hit_to_content(hit)

    @override
    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Content]:
        contents: list[Content] = []
        # TODO: Do this asynchronously?
        for id in set(ids):
            content = await self._aget_by_id_with_embedding(id)
            if content is not None:
                contents.append(content)
        return contents

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def _aget_by_id_with_embedding(self, document_id: str) -> Content | None:
        """
        Asynchronously retrieve a document by its ID, including its embedding.

        Parameters
        ----------
        document_id : str
            The document ID.

        Returns
        -------
        Content | None
            The retrieved document with embedding, or `None` if not found.
        """
        await self.vector_store.astra_env.aensure_db_setup()

        hit = await self.vector_store.astra_env.async_collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        return self._hit_to_content(hit)
