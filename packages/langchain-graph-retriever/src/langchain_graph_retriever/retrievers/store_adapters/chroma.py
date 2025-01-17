from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

from langchain_core.documents import Document

from .base import METADATA_EMBEDDING_KEY, StoreAdapter

try:
    from langchain_chroma import Chroma
except (ImportError, ModuleNotFoundError):
    msg = "please `pip install langchain-chroma`"
    raise ImportError(msg)


class ChromaStoreAdapter(StoreAdapter[Chroma]):
    def __init__(self, vector_store: Chroma):
        super().__init__(vector_store)

    @property
    def supports_normalized_metadata(self) -> bool:
        return False

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns docs (with embeddings) most similar to the query vector."""
        try:
            from chromadb.api.types import IncludeEnum
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install chromadb`"
            raise ImportError(msg)

        if k > self.vector_store._collection.count():
            k = self.vector_store._collection.count()

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
        for result in zip(
            results["documents"][0],  # type: ignore
            results["metadatas"][0],  # type: ignore
            results["ids"][0],  # type: ignore
            results["embeddings"][0],  # type: ignore
        ):
            metadata = result[1] or {}
            metadata[METADATA_EMBEDDING_KEY] = result[3]
            docs.append(
                Document(
                    page_content=result[0],
                    metadata=metadata,
                    id=result[2],
                )
            )
        return docs

    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Document]:
        """Get documents by id."""
        results = self.vector_store.get(ids=list(ids), **kwargs)
        return [
            Document(
                page_content=text,
                metadata=metadata,
                id=id,
            )
            for (text, metadata, id) in zip(
                results["documents"], results["metadatas"], results["ids"]
            )
        ]
