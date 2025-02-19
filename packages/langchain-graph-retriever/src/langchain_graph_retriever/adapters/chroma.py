"""Provides an adapter for Chroma vector store integration."""

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from typing_extensions import override

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter

try:
    from langchain_chroma import Chroma
except (ImportError, ModuleNotFoundError):
    msg = "please `pip install langchain-chroma`"
    raise ImportError(msg)


class ChromaAdapter(ShreddedLangchainAdapter[Chroma]):
    """
    Adapter for [Chroma](https://www.trychroma.com/) vector store.

    This adapter integrates the LangChain Chroma vector store with the
    graph retriever system, allowing for similarity search and document retrieval.

    Parameters
    ----------
    vector_store :
        The Chroma vector store instance.
    shredder: ShreddingTransformer, optional
        An instance of the ShreddingTransformer used for doc insertion.
        If not passed then a default instance of ShreddingTransformer is used.
    """

    @override
    def update_filter_hook(
        self, filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        filter = super().update_filter_hook(filter)
        if not filter or len(filter) <= 1:
            return filter

        conjoined = [{k: v} for k, v in filter.items()]
        return {"$and": conjoined}

    @override
    def _search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        try:
            from chromadb.api.types import IncludeEnum
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install chromadb`"
            raise ImportError(msg)

        if k > self.vector_store._collection.count():
            k = self.vector_store._collection.count()
        if k == 0:
            return []

        results = self.vector_store._collection.query(
            query_embeddings=embedding,  # type: ignore
            n_results=k,
            where=filter,  # type: ignore
            include=[
                IncludeEnum.documents,
                IncludeEnum.metadatas,
                IncludeEnum.embeddings,
            ],
            **kwargs,
        )

        docs: list[Document] = []
        # type-hint: (str, Dict[str, Any], str, ndarray)
        for content, metadata, id, emb in zip(
            results["documents"][0],  # type: ignore
            results["metadatas"][0],  # type: ignore
            results["ids"][0],  # type: ignore
            results["embeddings"][0],  # type: ignore
        ):
            docs.append(
                Document(
                    id=id,
                    page_content=content,
                    metadata={
                        METADATA_EMBEDDING_KEY: emb.tolist(),
                        **metadata,
                    },
                )
            )
        return docs

    @override
    def _get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        results = self.vector_store.get(
            ids=list(ids),
            include=["embeddings", "metadatas", "documents"],
            where=self.update_filter_hook(filter),
            **kwargs,
        )
        docs = [
            Document(
                id=id,
                page_content=content,
                metadata={METADATA_EMBEDDING_KEY: emb.tolist(), **metadata},
            )
            for (content, metadata, id, emb) in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
                results["embeddings"],
            )
            # type-hint: (str, Dict[str, Any], str, ndarray)
        ]

        return docs
