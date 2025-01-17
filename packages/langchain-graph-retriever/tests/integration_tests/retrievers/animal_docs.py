import json
import os

import pytest
from langchain_core.documents import Document
from tests.embeddings import AnimalEmbeddings
from tests.integration_tests.stores import StoreAdapter, StoreFactory


@pytest.fixture(scope="session")
def animal_docs() -> list[Document]:
    documents = []

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../../data/animals.jsonl")
    )
    with open(path, "r") as file:
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
def animal_store(
    request: pytest.FixtureRequest,
    store_factory: StoreFactory,
    animal_docs: list[Document],
) -> StoreAdapter:
    return store_factory.create(request, AnimalEmbeddings(), animal_docs)


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]
