import pytest
from langchain_core.documents import Document

from graph_pancake.retrievers.store_adapters import StoreAdapter
from tests.embeddings.simple_embeddings import EarthEmbeddings
from tests.integration_tests.stores import StoreFactory


@pytest.fixture(scope="session")
def earth_docs() -> list[Document]:
    """This is a set of Documents to pre-populate a graph vector store."""

    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )
    return [greetings, doc1, doc2]


@pytest.fixture(scope="session")
def earth_store(
    request: pytest.FixtureRequest,
    store_factory: StoreFactory,
    earth_docs: list[Document],
) -> StoreAdapter:
    return store_factory.create(request, EarthEmbeddings(), earth_docs)
