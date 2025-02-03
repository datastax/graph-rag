import json

import requests
from langchain_core.documents import Document

ANIMALS_JSONL_URL = "https://raw.githubusercontent.com/datastax/graph-rag/refs/heads/main/data/animals.jsonl"


def fetch_documents() -> list[Document]:
    """
    Download and parse a list of Documents for use with Graph Retriever.

    This is a small example dataset with useful links.

    This method downloads the dataset each time -- generally it is preferable
    to invoke this only once and store the documents in memory or a vector
    store.

    Returns
    -------
    :
        The fetched animal documents.
    """
    response = requests.get(ANIMALS_JSONL_URL)
    response.raise_for_status()  # Ensure we got a valid response

    return [
        Document(id=data["id"], page_content=data["text"], metadata=data["metadata"])
        for line in response.text.splitlines()
        if (data := json.loads(line))
    ]
