import asyncio
import json
import os

import pytest
from graph_retriever.testing.embeddings import AnimalEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pytest import Parser

from tests.embeddings import BaseEmbeddings


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
    return BaseEmbeddings(AnimalEmbeddings())
