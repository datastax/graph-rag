from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.retrievers.consts import METADATA_EMBEDDING_KEY

from .mmr_traversal_adapter import MMRTraversalAdapter


class OpenSearchMMRTraversalAdapter(MMRTraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_community.vectorstores import OpenSearchVectorSearch
        except (ImportError, ModuleNotFoundError):
            raise ImportError("please `pip install langchain-community opensearch-py`")

        self._base_vector_store = vector_store
        self._vector_store = cast(OpenSearchVectorSearch, vector_store)
        if self._vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for MMR Traversal: '{self._vector_store.engine}'"
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

    def similarity_search_with_embedding_by_vector(
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

        docs = self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            metadata_field="*",
            **kwargs,
        )

        # if metadata=="*" on the search, then the document
        # embedding vector and text are included in the
        # document metadata in the returned document.
        #
        # The actual document metadata is moved down into a
        # sub "metadata" key.
        for doc in docs:
            embedding = doc.metadata["vector_field"]
            doc.metadata = doc.metadata["metadata"] or {}
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding

        return docs

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install opensearch-py`."
            raise ImportError(msg)

        docs: list[Document] = []
        for id in ids:
            try:
                hit = self._vector_store.client.get(
                    index=self._vector_store.index_name,
                    id=id,
                    _source_includes=["text", "metadata"],
                    **kwargs,
                )
                docs.append(
                    Document(
                        page_content=hit["_source"]["text"],
                        metadata=hit["_source"]["metadata"],
                        id=hit["_id"],
                    )
                )
            except NotFoundError:
                pass
        return docs
