import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.adapters import Adapter

from tests.embeddings import AnimalEmbeddings
from tests.integration_tests.invoker import invoker

# Imports for definitions.
from tests.integration_tests.stores import (
    AdapterFactory,
    adapter_factory,
    enabled_stores,
    store_param,
)

# Mark these imports as used so they don't removed.
# They need to be imported here so the fixtures are available.
_ = (
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
