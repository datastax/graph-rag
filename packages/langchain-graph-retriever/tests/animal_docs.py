import json
import os

import pytest
from langchain_core.documents import Document


def load_animal_docs() -> list[Document]:
    documents = []

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../data/animals.jsonl")
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
def animal_docs() -> list[Document]:
    return load_animal_docs()


ANIMALS_QUERY: str = "small agile mammal"
ANIMALS_DEPTH_0_EXPECTED: list[str] = ["fox", "mongoose"]
