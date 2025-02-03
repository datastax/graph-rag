"""Provides an adapter for AstraDB vector store integration."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, cast

import backoff
from graph_retriever.content import Content
from graph_retriever.types import Edge, IdEdge, MetadataEdge
from graph_retriever.utils.batched import batched
from typing_extensions import override

try:
    from langchain_astradb import AstraDBVectorStore
    from langchain_astradb.utils.vector_store_codecs import (
        _AstraDBVectorStoreDocumentCodec,
    )
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-astradb`")

try:
    import astrapy
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install astrapy")
import httpx
from graph_retriever.adapters import Adapter
from langchain_core.documents import Document

_EXCEPTIONS_TO_RETRY = (
    httpx.TransportError,
    astrapy.exceptions.DataAPIException,
)
_MAX_RETRIES = 3


class _QueryHelper:
    def __init__(
        self,
        codec: _AstraDBVectorStoreDocumentCodec,
        user_filter: dict[str, Any] | None = {},
    ) -> None:
        self.codec = codec
        self.encoded_user_filter = (
            codec.encode_filter(user_filter) if user_filter else {}
        )

    def encode_filter(self, filter: dict[str, Any]) -> dict[str, Any]:
        encoded_filter = self.codec.encode_filter(filter)
        return self._with_user_filter(encoded_filter)

    def _with_user_filter(self, encoded_filter: dict[str, Any]) -> dict[str, Any]:
        if self.encoded_user_filter:
            return {"$and": [encoded_filter, self.encoded_user_filter]}
        else:
            return encoded_filter

    def create_ids_query(self, ids: list[str]) -> dict[str, Any]:
        if len(ids) == 0:
            raise ValueError("IDs should not be empty")
        elif len(ids) == 1:
            # We need to avoid re-encoding in this case, since the codecs
            # don't recognize `_id` and get confused (creating `metadata._id`).
            return self._with_user_filter({"_id": ids[0]})
        elif len(ids) < 100:
            return self._with_user_filter({"_id": {"$in": ids}})
        else:
            raise ValueError("IDs should be less than 100, was {len(ids)}")

    def create_metadata_query(
        self, metadata: dict[str, Iterable[Any]]
    ) -> dict[str, Any] | None:
        metadata = {k: v for k, v in metadata.items() if v}
        if not metadata:
            return None

        parts = []
        for k, v in metadata.items():
            # If there are more than 100 values, we can't create a single `$in` query.
            # But, we can do it for each batch of 100.
            for v_batch in batched(v, 100):
                batch = list(v_batch)
                if len(batch) == 1:
                    parts.append({k: batch[0]})
                else:
                    parts.append({k: {"$in": batch}})

        if len(parts) == 1:
            return self.encode_filter(parts[0])
        else:
            return self.encode_filter({"$or": parts})


class AstraAdapter(Adapter):
    """
    Adapter for AstraDBVectorStore.

    This adapter provides DataStax AstraDB support for the graph retriever
    system, enabling similarity search and document retrieval.

    It supports normalized metadata (collections of values) without
    denormalization.

    Parameters
    ----------
    vector_store : AstraDBVectorStore
        The AstraDB vector store instance.
    """

    def __init__(self, vector_store: AstraDBVectorStore) -> None:
        self.vector_store = vector_store

    @override
    def embed_query(self, query: str) -> list[float]:
        embedding = self.vector_store.embedding
        assert embedding is not None

        return embedding.embed_query(query)

    def _build_contents(
        self, docs_with_embeddings: list[tuple[Document, list[float]]]
    ) -> list[Content]:
        contents = []
        for doc, embedding in docs_with_embeddings:
            assert doc.id is not None
            contents.append(
                Content(
                    id=doc.id,
                    content=doc.page_content,
                    metadata=doc.metadata,
                    embedding=embedding,
                )
            )
        return contents

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding, docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return query_embedding, self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self.vector_store.asimilarity_search_with_embedding(
            query=query,
            k=k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_contents(docs_with_embeddings)

    @override
    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        docs_with_embeddings = (
            await self.vector_store.asimilarity_search_with_embedding_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        )
        return self._build_contents(docs_with_embeddings)

    @override
    def get(self, ids: Sequence[str], /, **kwargs: Any) -> list[Content]:
        contents: list[Content] = []
        for id in set(ids):
            content = self._get_by_id_with_embedding(id)
            if content is not None:
                contents.append(content)
        return contents

    def _hit_to_content(self, hit: dict[str, Any] | None) -> Content | None:
        if hit is None:
            return None
        doc = self.vector_store.document_codec.decode(hit)
        if doc is None:
            return None
        assert doc.id is not None
        embedding = self.vector_store.document_codec.decode_vector(hit)
        assert embedding is not None
        return Content(
            id=doc.id,
            content=doc.page_content,
            metadata=doc.metadata,
            embedding=embedding,
        )

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def _get_by_id_with_embedding(self, document_id: str) -> Content | None:
        """
        Retrieve a document by its ID, including its embedding.

        Parameters
        ----------
        document_id : str
            The document ID.

        Returns
        -------
        Content | None
            The retrieved document with embedding, or `None` if not found.
        """
        self.vector_store.astra_env.ensure_db_setup()

        hit = self.vector_store.astra_env.collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        return self._hit_to_content(hit)

    @override
    async def aget(self, ids: Sequence[str], /, **kwargs: Any) -> list[Content]:
        contents: list[Content] = []
        # TODO: Do this asynchronously?
        for id in set(ids):
            content = await self._aget_by_id_with_embedding(id)
            if content is not None:
                contents.append(content)
        return contents

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def _aget_by_id_with_embedding(self, document_id: str) -> Content | None:
        """
        Asynchronously retrieve a document by its ID, including its embedding.

        Parameters
        ----------
        document_id : str
            The document ID.

        Returns
        -------
        Content | None
            The retrieved document with embedding, or `None` if not found.
        """
        await self.vector_store.astra_env.aensure_db_setup()

        hit = await self.vector_store.astra_env.async_collection.find_one(
            {"_id": document_id},
            projection=self.vector_store.document_codec.full_projection,
        )
        return self._hit_to_content(hit)

    def _prepare_edge_query_parts(
        self, edges: set[Edge]
    ) -> tuple[dict[str, Iterable[Any]], set[str]]:
        """
        Return metadata and ID query parts for edges.

        Parameters
        ----------
        edges : set[Edge]
            The edges to prepare.

        Returns
        -------
        metadata_in : dict[str, set[Any]]
            Dictionary of metadata constraints indicating the field to constrain
            and the set of values.
        ids : set[str]
            Set of IDs to query for.

        Raises
        ------
        ValueError
            If any edges are invalid.
        """
        metadata_in: dict[str, set[Any]] = {}
        ids = set()

        for edge in edges:
            if isinstance(edge, MetadataEdge):
                metadata_in.setdefault(edge.incoming_field, set()).add(edge.value)
            elif isinstance(edge, IdEdge):
                ids.add(edge.id)
            else:
                raise ValueError(f"Unsupported edge {edge}")

        return (cast(dict[str, Iterable[Any]], metadata_in), ids)

    @override
    def get_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        query_helper = _QueryHelper(self.vector_store.document_codec, filter)
        metadata, ids = self._prepare_edge_query_parts(edges)

        results = []
        if (metadata_query := query_helper.create_metadata_query(metadata)) is not None:
            results.extend(
                self._execute_query(
                    k=k,
                    query_embedding=query_embedding,
                    query=metadata_query,
                )
            )
            ids.difference_update({c.id for c in results})

        for id_batch in batched(ids, 100):
            results.extend(
                self._execute_query(
                    k=k,
                    query_embedding=query_embedding,
                    query=query_helper.create_ids_query(list(id_batch)),
                )
            )

        return results

    @override
    async def aget_adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        query_helper = _QueryHelper(self.vector_store.document_codec, filter)
        metadata, ids = self._prepare_edge_query_parts(edges)

        results = []
        if (metadata_query := query_helper.create_metadata_query(metadata)) is not None:
            results.extend(
                await self._aexecute_query(
                    k=k,
                    query_embedding=query_embedding,
                    query=metadata_query,
                )
            )
            ids.difference_update({c.id for c in results})

        for id_batch in batched(ids, 100):
            # I have tested (in the lazy graph rag example) doing these
            # two queries concurrently, but it hurt perfomance. Likely this
            # would depend on how many IDs the previous query eliminates from
            # this query.
            results.extend(
                await self._aexecute_query(
                    k=k,
                    query_embedding=query_embedding,
                    query=query_helper.create_ids_query(list(id_batch)),
                )
            )

        return results

    def _execute_query(
        self, k: int, query_embedding: list[float], query: dict[str, Any]
    ) -> list[Content]:
        astra_env = self.vector_store.astra_env
        astra_env.ensure_db_setup()

        hits = astra_env.collection.find(
            filter=query,
            projection=self.vector_store.document_codec.full_projection,
            limit=k,
            include_sort_vector=True,
            sort={"$vector": query_embedding},
        )

        return [content for hit in hits if (content := self._decode_hit(hit))]

    async def _aexecute_query(
        self, k: int, query_embedding: list[float], query: dict[str, Any]
    ) -> list[Content]:
        astra_env = self.vector_store.astra_env
        await astra_env.aensure_db_setup()

        hits = astra_env.async_collection.find(
            filter=query,
            projection=self.vector_store.document_codec.full_projection,
            limit=k,
            include_sort_vector=True,
            sort={"$vector": query_embedding},
        )

        return [content async for hit in hits if (content := self._decode_hit(hit))]

    def _decode_hit(self, hit: dict[str, Any]) -> Content | None:
        codec = self.vector_store.document_codec
        if "metadata" not in hit or codec.content_field not in hit:
            id = hit.get("_id", "(no _id)")
            warnings.warn(
                f"Ignoring document with _id = {id}. Reason: missing required fields."
            )
            return None
        embedding = self.vector_store.document_codec.decode_vector(hit)
        assert embedding is not None
        return Content(
            id=hit["_id"],
            content=hit[codec.content_field],
            embedding=embedding,
            metadata=hit["metadata"],
        )
