"""Provides an adapter for OpenSearch vector store integration."""

from collections.abc import Sequence
from typing import Any

from typing_extensions import override

try:
    from langchain_community.vectorstores.opensearch_vector_search import (
        OpenSearchVectorSearch,
    )
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-community opensearch-py`")

from langchain_core.documents import Document

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import LangchainAdapter


class OpenSearchAdapter(LangchainAdapter[OpenSearchVectorSearch]):
    """
    Adapter to traverse OpenSearch vector stores.

    This adapter enables similarity search and document retrieval using an
    OpenSearch vector store.

    Parameters
    ----------
    vector_store  :
        The OpenSearch vector store instance.

    Notes
    -----
    Graph Traversal is only supported when using either the `"lucene"` or
    `"faiss"` engine.

    For more info, see the [OpenSearch Documentation](https://opensearch.org/docs/latest/search-plugins/knn/knn-index#method-definitions)
    """

    def __init__(self, vector_store: OpenSearchVectorSearch):
        if vector_store.engine not in ["lucene", "faiss"]:
            msg = (
                f"Invalid engine for Traversal: '{self.vector_store.engine}'"
                " please instantiate the Open Search Vector Store with"
                " either the 'lucene' or 'faiss' engine"
            )
            raise ValueError(msg)
        super().__init__(vector_store)

        if vector_store.is_aoss:
            self._id_field = "id"
        else:
            self._id_field = "_id"

    def _build_filter(
        self, filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]] | None:
        """
        Build a filter query for OpenSearch based on metadata.

        Parameters
        ----------
        filter :
            Metadata filter to apply.

        Returns
        -------
        :
            Filter query for OpenSearch.

        Raises
        ------
        ValueError
            If the query is not supported by OpenSearch adapter.
        """
        if filter is None:
            return None

        filters = []
        for key, value in filter.items():
            if isinstance(value, list):
                filters.append({"terms": {f"metadata.{key}": value}})
            elif isinstance(value, dict):
                raise ValueError("Open Search doesn't suport dictionary searches.")
            else:
                filters.append({"term": {f"metadata.{key}": value}})
        return filters

    @override
    def _search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        if filter is not None:
            # use an efficient_filter to collect results that
            # are near the embedding vector until up to 'k'
            # documents that match the filter are found.
            query = {"bool": {"must": self._build_filter(filter=filter)}}
            kwargs["efficient_filter"] = query

        if k == 0:
            return []

        docs = self.vector_store.similarity_search_by_vector(
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
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={
                    METADATA_EMBEDDING_KEY: doc.metadata["vector_field"],
                    **doc.metadata["metadata"],
                },
            )
            for doc in docs
        ]

    @override
    def _get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        query: dict[str, Any] = {"ids": {"values": ids}}

        if filter:
            query = {
                "bool": {"must": [query, *(self._build_filter(filter=filter) or [])]}
            }

        response = self.vector_store.client.search(
            body={
                "query": query,
            },
            index=self.vector_store.index_name,
            _source_includes=["text", "metadata", "vector_field"],
            **kwargs,
        )

        return [
            Document(
                page_content=hit["_source"]["text"],
                metadata={
                    METADATA_EMBEDDING_KEY: hit["_source"]["vector_field"],
                    **hit["_source"]["metadata"],
                },
                id=hit["_id"],
            )
            for hit in response["hits"]["hits"]
        ]
