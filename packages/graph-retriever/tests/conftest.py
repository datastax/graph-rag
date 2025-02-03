import asyncio

import pytest

pytest.register_assert_rewrite("graph_retriever.testing")

from tests.testing.adapters import animals  # noqa: E402
from tests.testing.invoker import sync_or_async  # noqa: E402

# Mark these imports as used so they don't get removed.
# They need to be imported in `conftest.py` so the fixtures are registered.
_ = (
    animals,
    sync_or_async,
)


# Event Loop for async tests.
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
