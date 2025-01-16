import json

import pytest
from langchain_core.documents import Document

from graph_pancake.document_transformers.metadata_denormalizer import (
    MetadataDenormalizer,
)
from tests.embeddings import AnimalEmbeddings
from tests.integration_tests.stores import StoreFactory, Stores


@pytest.fixture(scope="session")
def animal_docs() -> list[Document]:
    documents = []
    with open("tests/data/animals.jsonl", "r") as file:
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
    support_normalized_metadata: bool,
) -> Stores:
    if not support_normalized_metadata:
        animal_docs = list(MetadataDenormalizer().transform_documents(animal_docs))

    return store_factory.create(request, AnimalEmbeddings(), animal_docs)


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]
