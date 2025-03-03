from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import pytest
from graph_retriever import Node, atraverse, traverse
from graph_retriever.adapters import Adapter
from graph_retriever.edges import EdgeFunction, EdgeSpec
from graph_retriever.strategies import Strategy


class SyncOrAsync(abc.ABC):
    @abc.abstractmethod
    async def _traverse(self, **kwargs: Any) -> list[Node]: ...

    def traverse_sorted_ids(
        self,
        store: Adapter,
        query: str | None = None,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        strategy: Strategy | None = None,
    ) -> TraversalCall[list[str]]:
        return TraversalCall(
            transform=lambda nodes: sorted([n.id for n in nodes]),
            sync_or_async=self,
            store=store,
            query=query,
            edges=edges,
            strategy=strategy,
        )

    def traverse(
        self,
        store: Adapter,
        query: str | None = None,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        strategy: Strategy | None = None,
    ) -> TraversalCall[list[Node]]:
        return TraversalCall(
            transform=lambda nodes: nodes,
            sync_or_async=self,
            store=store,
            query=query,
            edges=edges,
            strategy=strategy,
        )


class SyncTraversal(SyncOrAsync):
    async def _traverse(self, **kwargs):
        return traverse(**kwargs)


class AsyncTraversal(SyncOrAsync):
    async def _traverse(self, **kwargs):
        return await atraverse(**kwargs)


T = TypeVar("T")


def _pick(name: str, call: T | None, init: T | None) -> T:
    value = call or init
    if value is None:
        raise ValueError(f"{name} must be set in call or init")
    return value


class TraversalCall(Generic[T]):
    def __init__(
        self,
        transform: Callable[[list[Node]], T],
        sync_or_async: SyncOrAsync,
        store: Adapter,
        query: str | None = None,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        strategy: Strategy | None = None,
    ) -> None:
        self.transform = transform
        self.sync_or_async = sync_or_async
        self.store = store
        self.query = query
        self.edges = edges
        self.strategy = strategy

    async def __call__(
        self,
        query: str | None = None,
        edges: list[EdgeSpec] | EdgeFunction | None = None,
        strategy: Strategy | None = None,
        metadata_filter: dict[str, Any] | None = None,
        initial_root_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> T:
        strategy = _pick("strategy", strategy, self.strategy)
        strategy = dataclasses.replace(strategy, **kwargs)

        results = await self.sync_or_async._traverse(
            query=_pick("query", query, self.query),
            store=self.store,
            edges=_pick("edges", edges, self.edges),
            strategy=strategy,
            metadata_filter=metadata_filter,
            initial_root_ids=initial_root_ids,
        )
        return self.transform(results)


@pytest.fixture(scope="function", params=["sync", "async"])
def sync_or_async(request: pytest.FixtureRequest) -> SyncOrAsync:
    if request.param == "sync":
        return SyncTraversal()
    elif request.param == "async":
        return AsyncTraversal()
    else:
        raise ValueError(f"Unexpected value '{request.param}'")
