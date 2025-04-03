"""Provides an adapter for AstraDB vector store integration."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, Literal, cast, overload

import backoff
from graph_retriever import Content
from graph_retriever.edges import Edge, IdEdge, MetadataEdge
from graph_retriever.utils import merge
from graph_retriever.utils.batched import batched
from graph_retriever.utils.top_k import top_k
from immutabledict import immutabledict
from typing_extensions import override

try:
    from langchain_astradb import AstraDBVectorStore
    from langchain_astradb.vectorstores import AstraDBQueryResult
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install langchain-astradb`")

try:
    import astrapy
except (ImportError, ModuleNotFoundError):
    raise ImportError("please `pip install astrapy")
import httpx
from graph_retriever.adapters import Adapter

_EXCEPTIONS_TO_RETRY = (
    httpx.TransportError,
    astrapy.exceptions.DataAPIException,
)
_MAX_RETRIES = 3


def _extract_queries(edges: set[Edge]) -> tuple[dict[str, Iterable[Any]], set[str]]:
    metadata: dict[str, set[Any]] = {}
    ids: set[str] = set()

    for edge in edges:
        if isinstance(edge, MetadataEdge):
            metadata.setdefault(edge.incoming_field, set()).add(edge.value)
        elif isinstance(edge, IdEdge):
            ids.add(edge.id)
        else:
            raise ValueError(f"Unsupported edge {edge}")

    return (cast(dict[str, Iterable[Any]], metadata), ids)


def _metadata_queries(
    user_filters: dict[str, Any] | None,
    metadata: dict[str, Iterable[Any]] = {},
) -> Iterator[dict[str, Any]]:
    """
    Generate queries for matching all user_filters and any `metadata`.

    The results of the queries can be merged to produce the results.

    Results will match at least one metadata value in one of the metadata fields.

    Results will also match all of the `user_filters`.

    Parameters
    ----------
    user_filters :
        User filters that all results must match.
    metadata :
        An item matches the queries if it matches all user filters, and
        there exists a `key` such that `metadata[key]` has a non-empty
        intersection with the actual values of `item.metadata[key]`.

    Yields
    ------
    :
        Queries corresponding to `user_filters AND metadata`.
    """
    if user_filters:

        def with_user_filters(filter: dict[str, Any]) -> dict[str, Any]:
            return {"$and": [filter, user_filters]}
    else:

        def with_user_filters(filter: dict[str, Any]) -> dict[str, Any]:
            return filter

    def process_value(v: Any) -> Any:
        if isinstance(v, immutabledict):
            return dict(v)
        else:
            return v

    for k, v in metadata.items():
        for v_batch in batched(v, 100):
            batch = [process_value(v) for v in v_batch]
            if isinstance(batch[0], dict):
                if len(batch) == 1:
                    yield with_user_filters({k: {"$all": [batch[0]]}})
                else:
                    yield with_user_filters(
                        {"$or": [{k: {"$all": [v]}} for v in batch]}
                    )
            else:
                if len(batch) == 1:
                    yield (with_user_filters({k: batch[0]}))
                else:
                    yield (with_user_filters({k: {"$in": batch}}))


async def empty_async_iterable() -> AsyncIterable[AstraDBQueryResult]:
    """Create an empty async iterable."""
    if False:
        yield


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

    def _build_content(self, result: AstraDBQueryResult) -> Content:
        assert result.embedding is not None
        return Content(
            id=result.id,
            content=result.document.page_content,
            metadata=result.document.metadata,
            embedding=result.embedding,
        )

    def _build_content_iter(
        self, results: Iterable[AstraDBQueryResult]
    ) -> Iterable[Content]:
        for result in results:
            yield self._build_content(result)

    async def _abuild_content_iter(
        self, results: AsyncIterable[AstraDBQueryResult]
    ) -> AsyncIterable[Content]:
        async for result in results:
            yield self._build_content(result)

    @overload
    def _run_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
    ) -> Iterable[Content]: ...

    @overload
    def _run_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[list[float], Iterable[Content]]: ...

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    def _run_query(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_sort_vector: bool = False,
    ) -> tuple[list[float], Iterable[Content]] | Iterable[Content]:
        if include_sort_vector:
            # Work around the fact that `k == 0` is rejected by Astra.
            # AstraDBVectorStore has a similar work around for non-vectorize path, but
            # we want it to apply in both cases.
            query_n = n if n > 0 else 1

            query_embedding, results = self.vector_store.run_query(
                n=query_n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_sort_vector=True,
                include_embeddings=True,
                include_similarity=False,
            )
            assert query_embedding is not None
            if n == 0:
                return query_embedding, self._build_content_iter([])
            return query_embedding, self._build_content_iter(results)
        else:
            results = self.vector_store.run_query(
                n=n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_sort_vector=False,
                include_embeddings=True,
                include_similarity=False,
            )
            return self._build_content_iter(results)

    @overload
    async def _arun_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
    ) -> AsyncIterable[Content]: ...

    @overload
    async def _arun_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
    ) -> tuple[list[float], AsyncIterable[Content]]: ...

    @backoff.on_exception(backoff.expo, _EXCEPTIONS_TO_RETRY, max_tries=_MAX_RETRIES)
    async def _arun_query(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_sort_vector: bool = False,
    ) -> tuple[list[float], AsyncIterable[Content]] | AsyncIterable[Content]:
        if include_sort_vector:
            # Work around the fact that `k == 0` is rejected by Astra.
            # AstraDBVectorStore has a similar work around for non-vectorize path, but
            # we want it to apply in both cases.
            query_n = n if n > 0 else 1

            query_embedding, results = await self.vector_store.arun_query(
                n=query_n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_sort_vector=True,
                include_embeddings=True,
                include_similarity=False,
            )
            assert query_embedding is not None
            if n == 0:
                return query_embedding, self._abuild_content_iter(
                    empty_async_iterable()
                )
            return query_embedding, self._abuild_content_iter(results)
        else:
            results = await self.vector_store.arun_query(
                n=n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_sort_vector=False,
                include_embeddings=True,
                include_similarity=False,
            )
            return self._abuild_content_iter(results)

    def _vector_sort_from_embedding(
        self,
        embedding: list[float],
    ) -> dict[str, Any]:
        return self.vector_store.document_codec.encode_vector_sort(vector=embedding)

    def _get_sort_and_optional_embedding(
        self, query: str, k: int
    ) -> tuple[None | list[float], dict[str, Any] | None]:
        if self.vector_store.document_codec.server_side_embeddings:
            sort = self.vector_store.document_codec.encode_vectorize_sort(query)
            return None, sort
        else:
            embedding = self.vector_store._get_safe_embedding().embed_query(query)
            if k == 0:
                return embedding, None  # signal that we should short-circuit
            sort = self._vector_sort_from_embedding(embedding)
            return embedding, sort

    @override
    def search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding, sort = self._get_sort_and_optional_embedding(query, k)
        if sort is None and query_embedding is not None:
            return query_embedding, []

        query_embedding, results = self._run_query(
            n=k, filter=filter, sort=sort, include_sort_vector=True
        )
        return query_embedding, list(results)

    @override
    async def asearch_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[Content]]:
        query_embedding, sort = self._get_sort_and_optional_embedding(query, k)
        if sort is None and query_embedding is not None:
            return query_embedding, []

        query_embedding, results = await self._arun_query(
            n=k, filter=filter, sort=sort, include_sort_vector=True
        )
        return query_embedding, [r async for r in results]

    @override
    def search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        if k == 0:
            return []
        sort = self._vector_sort_from_embedding(embedding)
        results = self._run_query(n=k, filter=filter, sort=sort)
        return list(results)

    @override
    async def asearch(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Content]:
        if k == 0:
            return []
        sort = self._vector_sort_from_embedding(embedding)
        results = await self._arun_query(n=k, filter=filter, sort=sort)
        return [r async for r in results]

    @override
    def get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        results = self._run_query(n=len(ids), ids=list(ids), filter=filter)
        return list(results)

    @override
    async def aget(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Content]:
        results = await self._arun_query(n=len(ids), ids=list(ids), filter=filter)
        return [r async for r in results]

    @override
    def adjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        sort = self._vector_sort_from_embedding(query_embedding)
        metadata, ids = _extract_queries(edges)

        metadata_queries = _metadata_queries(user_filters=filter, metadata=metadata)

        results: dict[str, Content] = {}
        for metadata_query in metadata_queries:
            # TODO: Look at a thread-pool for this.
            for result in self._run_query(n=k, filter=metadata_query, sort=sort):
                results[result.id] = result

        for id_batch in batched(ids, 100):
            for result in self._run_query(
                n=k, ids=list(id_batch), filter=filter, sort=sort
            ):
                results[result.id] = result

        return top_k(results.values(), embedding=query_embedding, k=k)

    @override
    async def aadjacent(
        self,
        edges: set[Edge],
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
        **kwargs: Any,
    ) -> Iterable[Content]:
        sort = self._vector_sort_from_embedding(query_embedding)
        metadata, ids = _extract_queries(edges)

        metadata_queries = _metadata_queries(user_filters=filter, metadata=metadata)

        iterables = []
        for metadata_query in metadata_queries:
            iterables.append(
                await self._arun_query(n=k, filter=metadata_query, sort=sort)
            )
        for id_batch in batched(ids, 100):
            iterables.append(
                await self._arun_query(
                    n=k, ids=list(id_batch), filter=filter, sort=sort
                )
            )

        iterators: list[AsyncIterator[Content]] = [it.__aiter__() for it in iterables]

        results: dict[str, Content] = {}
        async for result in merge.amerge(*iterators):
            results[result.id] = result

        return top_k(results.values(), embedding=query_embedding, k=k)
