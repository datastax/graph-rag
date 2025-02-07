"""Provides an adapter for Cassandra vector store integration."""

from collections.abc import Sequence
from typing import Any

from typing_extensions import override

try:
    from langchain_community.vectorstores.cassandra import Cassandra
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-community cassio`")

from langchain_core.documents import Document

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter


class CassandraAdapter(ShreddedLangchainAdapter[Cassandra]):
    """
    Adapter for the [Apache Cassandra](https://cassandra.apache.org/) vector store.

    This class integrates the LangChain Cassandra vector store with the graph
    retriever system, providing functionality for similarity search and document
    retrieval.

    Parameters
    ----------
    vector_store :
        The Cassandra vector store instance.
    shredder: ShreddingTransformer, optional
        An instance of the ShreddingTransformer used for doc insertion.
        If not passed then a default instance of ShreddingTransformer is used.
    """

    @override
    def _search(  # type: ignore
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        results = self._similarity_search_with_embedding_id_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs,
        )
        docs: list[Document] = []
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
        kwargs: dict[str, Any] = {}
        if filter is not None:
            kwargs["metadata"] = filter
        if body_search is not None:
            kwargs["body_search"] = body_search

        if k == 0:
            return []

        hits = self.vector_store.table.ann_search(
            vector=embedding,
            n=k,
            **kwargs,
        )
        return [
            (
                self.vector_store._row_to_document(row=hit),
                hit["vector"],
                hit["row_id"],
            )
            for hit in hits
        ]

    @override
    async def _asearch(  # type: ignore
        self, **kwargs: Any
    ) -> list[Document]:
        results = (
            await self.vector_store.asimilarity_search_with_embedding_id_by_vector(
                **kwargs
            )
        )
        docs: list[Document] = []
        for doc, embedding, id in results:
            doc.metadata[METADATA_EMBEDDING_KEY] = embedding
            doc.id = id
            docs.append(doc)
        return docs

    @override
    def _get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        filter = self.update_filter_hook(filter)
        docs: list[Document] = []
        for id in ids:
            args: dict[str, Any] = {"row_id": id}
            if filter:
                args["metadata"] = filter
            row = self.vector_store.table.get(**args)
            if row is not None:
                docs.append(self._row_to_doc(row))
        return docs

    @override
    async def _aget(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        filter = self.update_filter_hook(filter)
        docs: list[Document] = []
        for id in ids:
            args: dict[str, Any] = {"row_id": id}
            if filter:
                args["metadata"] = filter
            row = await self.vector_store.table.aget(**args)
            if row is not None:
                docs.append(self._row_to_doc(row))
        return docs

    def _row_to_doc(self, row: Any) -> Document:
        return Document(
            id=row["row_id"],
            page_content=row["body_blob"],
            metadata={
                METADATA_EMBEDDING_KEY: row["vector"],
                **row["metadata"],
            },
        )
