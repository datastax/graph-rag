import abc
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore

StoreT = TypeVar("StoreT", bound=VectorStore)

METADATA_EMBEDDING_KEY = "__embedding"


class StoreAdapter(Generic[StoreT], abc.ABC):
    def __init__(self, vector_store: StoreT):
        self.vector_store = vector_store

    @property
    def _safe_embedding(self) -> Embeddings:
        if not self.vector_store.embeddings:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.vector_store.embeddings

    def similarity_search_with_embedding(
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
        docs = self.similarity_search_with_embedding_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, docs

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
        return await run_in_executor(
            None, self.similarity_search_with_embedding, query, k, filter, **kwargs
        )

    @abc.abstractmethod
    def similarity_search_with_embedding_by_vector(
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
        return await run_in_executor(
            None,
            self.similarity_search_with_embedding_by_vector,
            embedding,
            k,
            filter,
            **kwargs,
        )

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.
        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.
        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.
        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.
        Returns:
            List[Document]: List of documents that were found.
        """

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.
        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.
        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.
        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            kwargs: Additional keyword arguments. These are up to the implementation.
        Returns:
            List[Document]: List of documents that were found.
        """
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )
