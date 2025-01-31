import json
import os

import pytest
from graph_retriever import Content
from graph_retriever.adapters import Adapter
from graph_retriever.adapters.in_memory import InMemory
from graph_retriever.testing import embeddings


def load_animal_content(embedding: embeddings.AnimalEmbeddings) -> list[Content]:
    documents = []

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../data/animals.jsonl")
    )
    with open(path) as file:
        for line in file:
            data = json.loads(line.strip())
            documents.append(
                Content(
                    id=data["id"],
                    content=data["text"],
                    embedding=embedding(data["text"]),
                    metadata=data["metadata"],
                )
            )
    return documents


@pytest.fixture(scope="session")
def animals() -> Adapter:
    embedding = embeddings.AnimalEmbeddings()
    return InMemory(embedding, load_animal_content(embedding))


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]
