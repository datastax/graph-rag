import asyncio

import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.adapters import Adapter
from pytest import Parser

from tests.animal_docs import animal_docs
from tests.embeddings import AnimalEmbeddings
from tests.invoker import invoker
from tests.stores import (
    ALL_STORES,
    TESTCONTAINER_STORES,
    AdapterFactory,
    adapter_factory,
    enabled_stores,
    store_param,
)

# Mark these imports as used so they don't removed.
# They need to be imported here so the fixtures are available.
_ = (
    animal_docs,
    adapter_factory,
    store_param,
    enabled_stores,
    invoker,
)


@pytest.fixture(scope="session")
def animal_store(
    request: pytest.FixtureRequest,
    adapter_factory: AdapterFactory,
    animal_docs: list[Document],
) -> Adapter:
    return adapter_factory.create(
        request, AnimalEmbeddings(), animal_docs, nested_metadata_fields={"keywords"}
    )


# may need to do some trickery if/when this is going to be used in end-user tests.
pytest.register_assert_rewrite("test.adapters.test_adapters")


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--stores",
        action="append",
        metavar="STORE",
        choices=ALL_STORES + ["all"],
        help="run tests for the given store (default: 'mem')",
    )
    parser.addoption(
        "--testcontainer",
        action="append",
        metavar="STORE",
        choices=TESTCONTAINER_STORES + ["none"],
        help="which stores to run testcontainers for (default: 'all')",
    )


# TODO: Allow marking tests as only supporting a subset of stores?
#
# def pytest_configure(config):
#     # register an additional marker
#     config.addinivalue_line(
#         "markers", "svc(name): tests that require the named service"
#     )
#
# def pytest_runtest_setup(item):
#     """Skip the test unless all of the marked services are present."""
#
#     required_svcs = {mark.args[0] for mark in item.iter_markers(name="svc")}
#     provided_svcs = set(item.config.getoption("-S") or [])
#
#     missing_svcs = required_svcs - provided_svcs
#     if missing_svcs:
#         pytest.skip(f"test requires services {missing_svcs!r}")
