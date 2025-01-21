from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    override,
)

from langchain_core.documents import Document

from .base import METADATA_EMBEDDING_KEY, Adapter

try:
    from langchain_chroma import Chroma
except (ImportError, ModuleNotFoundError):
    msg = "please `pip install langchain-chroma`"
    raise ImportError(msg)


class ChromaAdapter(Adapter[Chroma]):
    """Adapter Chroma vector store."""

    def __init__(
        self,
        vector_store: Chroma,
        *,
        denormalized_path_delimiter: str = ".",
        denormalized_static_value: str = "$",
    ):
        super().__init__(
            vector_store,
            use_normalized_metadata=False,
            denormalized_path_delimiter=denormalized_path_delimiter,
            denormalized_static_value=denormalized_static_value,
        )

    @override
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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
                        METADATA_EMBEDDING_KEY: emb,
                        **metadata,
                    },
                )
            )
        return docs

    @override
    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        results = self.vector_store.get(
            ids=list(ids), include=["embeddings", "metadatas", "documents"], **kwargs
        )
        return [
            Document(
                id=id,
                page_content=content,
                metadata={METADATA_EMBEDDING_KEY: emb, **metadata},
            )
            for (content, metadata, id, emb) in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
                results["embeddings"],
            )
        ]
