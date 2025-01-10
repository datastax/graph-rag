from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore


class TraversalAdapter:
    _base_vector_store: VectorStore

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self._base_vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self._base_vector_store.embeddings

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Return docs most similar to query.

        Also returns the embedded query vector.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        query_embedding = self._safe_embedding.embed_query(text=query)
        docs = self.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Document]]:
        """Return docs most similar to query.

        Also returns the embedded query vector.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.
        """
        return await run_in_executor(
            None, self.similarity_search, query, k, filter, **kwargs
        )

    @abstractmethod
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Async return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query vector.
        """
        return await run_in_executor(
            None,
            self.similarity_search_by_vector,
            embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
