import asyncio
import json
import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pytest import Parser

from tests.embeddings import BaseEmbeddings

pytest.register_assert_rewrite("graph_retriever.testing")


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


ALL_STORES = ["mem", "opensearch", "astra", "cassandra", "chroma"]
TESTCONTAINER_STORES = ["opensearch", "cassandra"]


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
    parser.addoption(
        "--runextras", action="store_true", default=False, help="run tests for extras"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "extra: mark test as requiring an `extra` package"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runextras"):
        # --runextras given in cli: do not skip extras
        return
    skip_extras = pytest.mark.skip(reason="need --runextras option to run")
    for item in items:
        if "extra" in item.keywords:
            item.add_marker(skip_extras)


@pytest.fixture(scope="session")
def enabled_stores(request: pytest.FixtureRequest) -> set[str]:
    # TODO: Use StrEnum?
    stores = request.config.getoption("--stores")

    if stores and "all" in stores:
        return set(ALL_STORES)
    elif stores:
        return set(stores)
    else:
        return {"mem"}


@pytest.fixture(scope="session")
def testcontainers(request: pytest.FixtureRequest) -> set[str]:
    # TODO: Use StrEnum?
    testcontainers = set(request.config.getoption("--testcontainer") or [])
    if testcontainers and "none" in testcontainers:
        return set()
    elif not testcontainers:
        return set(TESTCONTAINER_STORES)
    else:
        return testcontainers


def load_animal_docs() -> list[Document]:
    documents = []

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data/animals.jsonl")
    )
    with open(path) as file:
        for line in file:
            data = json.loads(line.strip())
            documents.append(
                Document(
                    id=data["id"],
                    page_content=data["text"],
                    metadata=data["metadata"],
                )
            )
    return documents


@pytest.fixture(scope="session")
def animal_docs() -> list[Document]:
    return load_animal_docs()


@pytest.fixture(scope="session")
def animal_embeddings() -> Embeddings:
    # This must be imported late (after registering the rewrites)
    from graph_retriever.testing.embeddings import AnimalEmbeddings

    return BaseEmbeddings(AnimalEmbeddings())
