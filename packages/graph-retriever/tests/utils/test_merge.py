import asyncio
from collections.abc import AsyncIterator

import pytest
from graph_retriever.utils.merge import amerge


async def async_generator(
    values: list[int], event: asyncio.Event | None = None
) -> AsyncIterator[int]:
    """Helper async generator that yields values with an optional delay."""
    for value in values:
        if event:
            await event.wait()
            event.clear()
        yield value


async def test_amerge_basic():
    """Test merging two basic async iterators."""
    gen1 = async_generator([1, 3, 5])
    gen2 = async_generator([2, 4, 6])

    result = [val async for val in amerge(gen1, gen2)]
    assert sorted(result) == [1, 2, 3, 4, 5, 6]


async def test_merge_empty_iterators():
    """Test merging when one of the iterators is empty."""
    gen1 = async_generator([])
    gen2 = async_generator([1, 2, 3])

    result = [val async for val in amerge(gen1, gen2)]
    assert result == [1, 2, 3]  # Should return only the non-empty iterator's items


async def test_merge_all_empty():
    """Test merging when all iterators are empty."""
    gen1 = async_generator([])
    gen2 = async_generator([])

    result = [val async for val in amerge(gen1, gen2)]
    assert result == []  # Should return an empty list


async def test_merge_large_iterators():
    """Test merging large iterators."""
    gen1 = async_generator(range(100))
    gen2 = async_generator(range(100, 200))

    result = [val async for val in amerge(gen1, gen2)]
    result.sort()
    assert result == list(range(200))  # Ensure all items are included


async def test_merge_unordered_iterators():
    """Ensure iterators are merged in order of availability, not sorting."""
    e1 = asyncio.Event()
    e2 = asyncio.Event()
    gen1 = async_generator([10, 30, 50], e1)
    gen2 = async_generator([20, 40], e2)

    it = amerge(gen1, gen2)
    e1.set()
    assert await anext(it) == 10
    e1.set()
    assert await anext(it) == 30
    e1.set()
    assert await anext(it) == 50
    e1.set()
    e2.set()
    assert await anext(it) == 20
    e2.set()
    assert await anext(it) == 40
    e1.set()
    e2.set()
    assert await anext(it, None) is None


async def test_merge_exception_handling():
    """Ensure that an exception in one iterator does not break everything."""

    async def faulty_generator():
        """Async generator that raises an exception mid-way."""
        yield 1
        yield 2
        raise ValueError("Test exception")
        yield 3  # Should never be reached

    gen1 = async_generator([10, 20, 30])
    gen2 = faulty_generator()

    with pytest.raises(ValueError, match="Test exception"):
        _result = [val async for val in amerge(gen1, gen2)]
