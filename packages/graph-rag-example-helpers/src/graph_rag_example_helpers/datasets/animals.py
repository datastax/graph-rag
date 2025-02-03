import json

import requests
from langchain_core.documents import Document

ANIMALS_JSONL_URL = "https://raw.githubusercontent.com/datastax/graph-rag/refs/heads/main/data/animals.jsonl"


class Animals:
    """Download a list of Documents for experimenting with Graph-Retriever."""

    def __init__(self):
        response = requests.get(ANIMALS_JSONL_URL)
        response.raise_for_status()  # Ensure we got a valid response

        self.documents = []
        for line in response.text.splitlines():
            data = json.loads(line)  # Parse each line as JSON
            self.documents.append(
                Document(
                    id=data["id"], page_content=data["text"], metadata=data["metadata"]
                )
            )

    def docs(self) -> list[Document]:
        """Get a list of Documents for experimenting with Graph-Retriever."""
        return self.documents
