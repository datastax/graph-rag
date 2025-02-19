"""Provides an adapter for AstraDB vector store integration."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

import backoff
from graph_retriever import Content
from graph_retriever.edges import Edge, IdEdge, MetadataEdge
from graph_retriever.utils import merge
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


def _extract_queries(edges: set[Edge]) -> tuple[dict[str, Iterable[Any]], Sequence[dict[str, Any]], set[str]]:
    # This combines queries for single fields -- `{"a": 5}` and `{"a": 7}`
    # becomes `{"a": {5, 7}}` allowing for an efficient `$in` query.
    #
    # It does so only for edges with single field constraints. In *theory* it
    # could combine cases where all the *other* constraints were equal -- `{"a":
    # 5, "b": 3}` and `{"a": 5, "b" 7}` could become `{"a": 5, "b": {"$in": [3,
    # 7]}}` but that would require deeper analysis. It is a possible future
    # improvement. To make that efficient, we may wish to know which field is
    # more "varying" (e.g, if one field is "entity type" from a small enum and
    # the other is "entity label" from a large set of values), then we could
    # collect constraints with the entity type in common. It may be possible to
    # encode this information in the edge itself, so the edge function is
    # responsible for guiding the combining.
    single_metadata: dict[str, set[Any]] = {}
    multi_metadata: list[dict[str, Any]] = []
    ids: set[str] = set()

    for edge in edges:
        if isinstance(edge, MetadataEdge):
            if len(edge.fields) == 1:
                k, v = next(iter(edge.fields.items()))
                single_metadata.setdefault(k, set()).add(v)
            else:
                multi_metadata.append(edge.fields)
        elif isinstance(edge, IdEdge):
            ids.add(edge.id)
        else:
            raise ValueError(f"Unsupported edge {edge}")

    return (cast(dict[str, Iterable[Any]], single_metadata),
            cast(Sequence[dict[str, Any]], multi_metadata),
            ids)


def _queries(
    codec: _AstraDBVectorStoreDocumentCodec,
    user_filters: dict[str, Any] | None,
    *,
    single_metadata: dict[str, Iterable[Any]] = {},
    multi_metadata: Sequence[dict[str, Any]] = (),
    ids: Iterable[str] = (),
) -> Iterator[dict[str, Any]]:
    """
    Generate queries for matching all user_filters and any `metadata`.

    The results of the queries can be merged to produce the results.

    Results will match at least one metadata value in one of the metadata fields
    or one of the IDs.

    Results will also match all of the `user_filters`.

    Parameters
    ----------
    codec :
        Codec to use for encoding the queries.
    user_filters :
        User filters that all results must match.
    single_metadata :
        An item matches the queries if it matches all user filters, and
        there exists a `key` such that `metadata[key]` has a non-empty
        intersection with the actual values of `item.metadata[key]`.

        These are collected from metadata edges affecting the single fields,
        which allows a simpler implementaiton of `ANY` by simply using a `$in`.
    multi_metadata :
        An item matches the queries if it matches all user filters and any
        of the filters in this list.
    ids :
        An item matches the queries if it matches all user filters, and
        it has an `item.id` in `ids`.

    Yields
    ------
    :
        Queries corresponding to `user_filters AND (metadata OR ids)`.
    """
    if user_filters:
        encoded_user_filters = codec.encode_filter(user_filters) if user_filters else {}

        def with_user_filters(
            filter: dict[str, Any], *, encoded: bool
        ) -> dict[str, Any]:
            return {
                "$and": [
                    filter if encoded else codec.encode_filter(filter),
                    encoded_user_filters,
                ]
            }
    else:

        def with_user_filters(
            filter: dict[str, Any], *, encoded: bool
        ) -> dict[str, Any]:
            return filter if encoded else codec.encode_filter(filter)

    for k, v in single_metadata.items():
        for v_batch in batched(v, 100):
            batch = list(v_batch)
            if len(batch) == 1:
                yield with_user_filters({k: batch[0]}, encoded=False)
            else:
                yield with_user_filters({k: {"$in": batch}}, encoded=False)

    for filter_batch in batched(multi_metadata, 100):
        def rewrite_multi(multi: dict[str, Any]) -> dict[str, Any]:
            if len(multi) == 1:
                return multi
            else:
                return {"$and": [{k: v} for k, v in multi.items()]}
        batch = [rewrite_multi(f) for f in filter_batch]
        if len(batch) == 1:
            yield with_user_filters(batch[0], encoded=False)
        else:
            yield with_user_filters({"$or": batch}, encoded=False)

    for id_batch in batched(ids, 100):
        ids = list(id_batch)
        if len(ids) == 1:
            yield with_user_filters({"_id": ids[0]}, encoded=True)
        else:
            assert len(ids) > 1 and len(ids) <= 100
            yield with_user_filters({"_id": {"$in": ids}}, encoded=True)


class AstraAdapter(Adapter):
    """
    Adapter for the [AstraDB](https://www.datastax.com/products/datastax-astra) vector store.

    This class integrates the LangChain AstraDB vector store with the graph
    retriever system, providing functionality for similarity search and document
    retrieval.

    Parameters
    ----------
    vector_store :
        The AstraDB vector store instance.
    """  # noqa: E501

    def __init__(self, vector_store: AstraDBVectorStore) -> None:
        self.vector_store = vector_store.copy(
            component_name="langchain_graph_retriever"
        )

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
        # Work around the fact that `k == 0` is rejected by Astra.
        # AstraDBVectorStore has a similar work around for non-vectorize path, but
        # we want it to apply in both cases.
        query_k = k
        if query_k == 0:
            query_k = 1
        query_embedding, docs_with_embeddings = (
            self.vector_store.similarity_search_with_embedding(
                query=query,
                k=query_k,
                filter=filter,
                **kwargs,
            )
        )
        if k == 0:
            return query_embedding, []

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
        # Work around the fact that `k == 0` is rejected by Astra.
        # AstraDBVectorStore has a similar work around for non-vectorize path, but
        # we want it to apply in both cases.
        query_k = k
        if query_k == 0:
            query_k = 1
        (
            query_embedding,
            docs_with_embeddings,
        ) = await self.vector_store.asimilarity_search_with_embedding(
            query=query,
            k=query_k,
            filter=filter,
            **kwargs,
        )
        if k == 0:
            return query_embedding, []
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
        if k == 0:
            return []

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
        if k == 0:
            return []

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
        return self._execute_and_merge(
            _queries(
                codec=self.vector_store.document_codec,
                user_filters=filter,
                ids=set(ids),
            )
        )

    @override
    async def aget(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        return await self._aexecute_and_merge(
            _queries(
                codec=self.vector_store.document_codec,
                user_filters=filter,
                ids=set(ids),
            )
        )

    @override
    def adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        sort = self.vector_store.document_codec.encode_vector_sort(query_embedding)
        single_metadata, multi_metadata, ids = _extract_queries(edges)
        if multi_metadata:
            print(f"MULTI: {multi_metadata}")
        filters = _queries(
            codec=self.vector_store.document_codec,
            user_filters=filter,
            single_metadata=single_metadata,
            multi_metadata=multi_metadata,
            ids=ids,
        )

        results = self._execute_and_merge(
            filters=filters,
            sort=sort,
            limit=k,
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
        sort = self.vector_store.document_codec.encode_vector_sort(query_embedding)
        single_metadata, multi_metadata, ids = _extract_queries(edges)
        filters = _queries(
            codec=self.vector_store.document_codec,
            user_filters=filter,
            single_metadata=single_metadata,
            multi_metadata=multi_metadata,
            ids=ids,
        )

        results = await self._aexecute_and_merge(
            filters=filters,
            sort=sort,
            limit=k,
        )
        return top_k(results, embedding=query_embedding, k=k)

    def _execute_and_merge(
        self,
        filters: Iterator[dict[str, Any]],
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

        results: dict[str, Content] = {}
        for filter in filters:
            print(f"FILTER: {filter}")
            # TODO: Look at a thread-pool for this.
            hits = astra_env.collection.find(
                filter=filter,
                projection=self.vector_store.document_codec.full_projection,
                limit=limit,
                include_sort_vector=True,
                include_similarity=include_similarity,
                sort=sort,
            )

            for hit in hits:
                if (
                    hit["_id"] not in results
                    and (content := self._decode_hit(hit)) is not None
                ):
                    results[content.id] = content

        return list(results.values())

    async def _aexecute_and_merge(
        self,
        filters: Iterator[dict[str, Any]],
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

        cursors = []
        for filter in filters:
            cursors.append(
                astra_env.async_collection.find(
                    filter=filter,
                    limit=limit,
                    projection=self.vector_store.document_codec.full_projection,
                    include_sort_vector=True,
                    include_similarity=include_similarity,
                    sort=sort,
                )
            )

        results: dict[str, Content] = {}
        async for hit in merge.amerge(*cursors):
            if (
                hit["_id"] not in results
                and (content := self._decode_hit(hit)) is not None
            ):
                results[content.id] = content

        return list(results.values())

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
