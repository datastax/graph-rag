"""Provides an adapter for AstraDB vector store integration."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, cast

import backoff
from graph_retriever.content import Content
from graph_retriever.types import Edge, IdEdge, MetadataEdge
from graph_retriever.utils.batched import batched
from graph_retriever.utils.top_k import top_k
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
    Adapter for the [AstraDB](https://www.datastax.com/products/datastax-astra) vector store.

    This class integrates the LangChain AstraDB vector store with the graph
    retriever system, providing functionality for similarity search and document
    retrieval.

    Parameters
    ----------
    vector_store : AstraDBVectorStore
        The AstraDB vector store instance.
    """  # noqa: E501

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
    def search_with_embedding(
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
    async def asearch_with_embedding(
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
    def search(
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
    async def asearch(
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
    def get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        helper = _QueryHelper(self.vector_store.document_codec, filter)

        results: dict[str, Content] = {}
        for batch in batched(set(ids), 100):
            query = helper.create_ids_query(list(batch))

            # TODO: Consider deduplicating before decoding?
            for content in self._execute_query(query=query):
                results.setdefault(content.id, content)
        return list(results.values())

    @override
    async def aget(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        helper = _QueryHelper(self.vector_store.document_codec, filter)

        tasks = set()
        for batch in batched(set(ids), 100):
            query = helper.create_ids_query(list(batch))
            tasks.add(asyncio.create_task(self._aexecute_query(query=query)))

        results: dict[str, Content] = {}
        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            tasks = pending

            # TODO: Consider deduplicating before decoding?
            for contents in done:
                for content in await contents:
                    results.setdefault(content.id, content)

        return list(results.values())

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
    def adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        query_helper = _QueryHelper(self.vector_store.document_codec, filter)
        metadata, ids = self._prepare_edge_query_parts(edges)

        sort = self.vector_store.document_codec.encode_vector_sort(query_embedding)

        results = []
        if (metadata_query := query_helper.create_metadata_query(metadata)) is not None:
            batch = self._execute_query(
                limit=k,
                sort=sort,
                query=metadata_query,
            )
            results.extend(batch)
            ids.difference_update({c.id for c in batch})

        for id_batch in batched(ids, 100):
            results.extend(
                self._execute_query(
                    limit=k,
                    sort=sort,
                    query=query_helper.create_ids_query(list(id_batch)),
                )
            )

        return top_k(results, embedding=query_embedding, k=k)

    @override
    async def aadjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        query_helper = _QueryHelper(self.vector_store.document_codec, filter)
        metadata, ids = self._prepare_edge_query_parts(edges)

        sort = self.vector_store.document_codec.encode_vector_sort(query_embedding)

        results = []
        if (metadata_query := query_helper.create_metadata_query(metadata)) is not None:
            batch = await self._aexecute_query(
                limit=k,
                sort=sort,
                query=metadata_query,
            )
            results.extend(batch)
            ids.difference_update({c.id for c in batch})

        for id_batch in batched(ids, 100):
            # I have tested (in the lazy graph rag example) doing these
            # two queries concurrently, but it hurt perfomance. Likely this
            # would depend on how many IDs the previous query eliminates from
            # this query.
            result_batch = await self._aexecute_query(
                limit=k,
                sort=sort,
                query=query_helper.create_ids_query(list(id_batch)),
            )
            results.extend(result_batch)

        return top_k(results, embedding=query_embedding, k=k)

    def _execute_query(
        self,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        sort: dict[str, Any] | None = None,
    ) -> list[Content]:
        astra_env = self.vector_store.astra_env
        astra_env.ensure_db_setup()

        # Similarity can only be included if we are sorting by vector.
        # Ideally, we could request the `$similarity` projection even
        # without vector sort. And it can't be `False`. It needs to be
        # `None` or it will cause an assertion error.
        include_similarity = None
        if not (sort or {}).keys().isdisjoint({"$vector", "$vectorize"}):
            include_similarity = True
        hits = astra_env.collection.find(
            filter=query,
            projection=self.vector_store.document_codec.full_projection,
            limit=limit,
            include_sort_vector=True,
            include_similarity=include_similarity,
            sort=sort,
        )

        return [content for hit in hits if (content := self._decode_hit(hit))]

    async def _aexecute_query(
        self,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        sort: dict[str, Any] | None = None,
    ) -> list[Content]:
        astra_env = self.vector_store.astra_env
        await astra_env.aensure_db_setup()

        # Similarity can only be included if we are sorting by vector.
        # Ideally, we could request the `$similarity` projection even
        # without vector sort. And it can't be `False`. It needs to be
        # `None` or it will cause an assertion error.
        include_similarity = None
        if not (sort or {}).keys().isdisjoint({"$vector", "$vectorize"}):
            include_similarity = True
        hits = astra_env.async_collection.find(
            filter=query,
            projection=self.vector_store.document_codec.full_projection,
            limit=limit,
            include_sort_vector=True,
            include_similarity=include_similarity,
            sort=sort,
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
            # We may not have `$similarity` depending on the query.
            score=hit.get("$similarity", None),
        )
