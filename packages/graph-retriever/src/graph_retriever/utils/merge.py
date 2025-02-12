import asyncio
import dataclasses
from collections.abc import AsyncIterator
from typing import TypeVar

T = TypeVar("T")


@dataclasses.dataclass
class _Done:
    exception: bool = False


async def amerge(
    *async_iterables: AsyncIterator[T],
    queue_size: int = 10,
) -> AsyncIterator[T]:
    """
    Merge async iterables into a single async iterator.

    Elements are yielded in the order they become available.

    Parameters
    ----------
    async_iterables :
        The async iterators to merge.
    queue_size :
        Number of elements to buffer in the queue.

    Yields
    ------
    :
        The elements of the iterators as they become available.
    """
    queue: asyncio.Queue[T | _Done] = asyncio.Queue(queue_size)

    async def pump(aiter: AsyncIterator[T]) -> None:
        try:
            async for item in aiter:
                await queue.put(item)
            await queue.put(_Done(exception=False))
        except:
            await queue.put(_Done(exception=True))
            raise

    tasks = [asyncio.create_task(pump(aiter)) for aiter in async_iterables]

    try:
        pending_count = len(async_iterables)
        while pending_count > 0:
            item = await queue.get()
            if isinstance(item, _Done):
                if item.exception:
                    # If there has been an exception, end early.
                    break
                else:
                    pending_count -= 1
            else:
                yield item
            queue.task_done()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks)
