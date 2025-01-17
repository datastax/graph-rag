from typing import Any

import pytest
from langchain_core.retrievers import BaseRetriever


@pytest.fixture(scope="function", params=["sync", "async"])
def invoker(request: pytest.FixtureRequest):
    async def sync_invoker(
        retriever: BaseRetriever, input: str, *args: Any, **kwargs: Any
    ):
        return retriever.invoke(input, *args, **kwargs)

    async def async_invoker(
        retriever: BaseRetriever, input: str, *args: Any, **kwargs: Any
    ):
        return await retriever.ainvoke(input, *args, **kwargs)

    if request.param == "sync":
        return sync_invoker
    elif request.param == "async":
        return async_invoker
    else:
        raise ValueError(f"Unexpected value '{request.param}'")
