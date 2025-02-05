import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from contextvars import copy_context
from functools import partial
from typing import ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


async def run_in_executor(
    executor: Executor | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:  # noqa: DOC502
    """
    Run a function in an executor.

    Parameters
    ----------
    executor :
        The executor to run in.
    func :
        The function.
    *args :
        The positional arguments to the function.
    kwargs :
        The keyword arguments to the function.

    Returns
    -------
    :
        The output of the function.

    Raises
    ------
    RuntimeError
        If the function raises a StopIteration.
    """  # noqa: DOC502

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    if executor is None or isinstance(executor, dict):
        # Use default executor with context copied from current context
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast(Callable[..., T], partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor, wrapper)
