from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.retrievers.consts import METADATA_EMBEDDING_KEY

from .mmr_traversal_adapter import MMRTraversalAdapter


class CassandraMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_community.vectorstores import Cassandra
        except (ImportError, ModuleNotFoundError):
            raise ImportError("please `pip install langchain-community cassio`")

        self._vector_store = cast(Cassandra, vector_store)
        self._base_vector_store = vector_store

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Returns docs (with embeddings) most similar to the query.

        Also returns the embedded query vector.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of:
                * The embedded query vector
                * List of Documents most similar to the query vector.
                  Documents should have their embedding added to
                  their metadata under the METADATA_EMBEDDING_KEY key.
        """
        query_embedding = self._safe_embedding.embed_query(text=query)
        docs = await self.asimilarity_search_with_embedding_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    def similarity_search_with_embedding_by_vector(  # type: ignore
        self,
        embedding: List[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self._similarity_search_with_embedding_id_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        docs: List[Document] = []
        for doc, embedding, id in results:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            doc.id = id
            docs.append(doc)
        return docs

    def _similarity_search_with_embedding_id_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        body_search: str | list[str] | None = None,
    ) -> list[tuple[Document, list[float], str]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            body_search: Document textual search terms to apply.
                Only supported by Astra DB at the moment.
        Returns:
            List of (Document, embedding, id), the most similar to the query vector.
        """
        kwargs: dict[str, Any] = {}
        if filter is not None:
            kwargs["metadata"] = filter
        if body_search is not None:
            kwargs["body_search"] = body_search

        hits = self._vector_store.table.ann_search(
            vector=embedding,
            n=k,
            **kwargs,
        )
        return [
            (
                self._vector_store._row_to_document(row=hit),
                hit["vector"],
                hit["row_id"],
            )
            for hit in hits
        ]

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query vector.
                Documents should have their embedding added to
                their metadata under the METADATA_EMBEDDING_KEY key.
        """
        results = (
            await self._vector_store.asimilarity_search_with_embedding_id_by_vector(
                embedding=embedding, k=k, filter=filter, **kwargs
            )
        )
        docs: List[Document] = []
        for doc, embedding, id in results:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            doc.id = id
            docs.append(doc)
        return docs

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = self._vector_store.get_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs

    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        docs: list[Document] = []
        for id in ids:
            doc = await self._vector_store.aget_by_document_id(id, **kwargs)
            if doc is not None:
                docs.append(doc)
        return docs
