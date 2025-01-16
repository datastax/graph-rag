from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .traversal_adapter import TraversalAdapter


class OpenSearchTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_community.vectorstores import OpenSearchVectorSearch
        except (ImportError, ModuleNotFoundError):
            raise ImportError("please `pip install langchain-community opensearch-py`")

        self._base_vector_store = vector_store
        self._vector_store = cast(OpenSearchVectorSearch, vector_store)
        if self._vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for Traversal: '{self._vector_store.engine}'"
                " please instantiate the Open Search Vector Store with"
                " either the 'lucene' or 'faiss' engine"
            )
            raise ValueError(msg)

    def _build_filter(
        self, filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]] | None:
        if filter is None:
            return None
        return [
            {
                "terms" if isinstance(value, list) else "term": {
                    f"metadata.{key}.keyword": value
                }
            }
            for key, value in filter.items()
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector."""
        if filter is not None:
            # use an efficient_filter to collect results that
            # are near the embedding vector until up to 'k'
            # documents that match the filter are found.
            kwargs["efficient_filter"] = {
                "bool": {"must": self._build_filter(filter=filter)}
            }

        return self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            **kwargs,
        )
