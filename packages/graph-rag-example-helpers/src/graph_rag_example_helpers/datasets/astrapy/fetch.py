import json

import requests
from langchain_core.documents import Document

ASTRAPY_JSONL_URL = "https://raw.githubusercontent.com/datastax/graph-rag/refs/heads/main/data/astrapy.jsonl"


def fetch_documents() -> list[Document]:
    """
    Download and parse a list of Documents for use with Graph Retriever.

    This dataset contains the documentation for the AstraPy project as of version 1.5.2.

    This method downloads the dataset each time -- generally it is preferable
    to invoke this only once and store the documents in memory or a vector
    store.

    Returns
    -------
    :
        The fetched astra-py documentation Documents.

    Notes
    -----
    - The dataset is setup in a way where the path of the item is the `id`, the pydoc
    description is the `page_content`, and the items other attributes are stored in the
    `metadata`.
    - There are many documents that contain an id and metadata, but no page_content.
    """
    response = requests.get(ASTRAPY_JSONL_URL)
    response.raise_for_status()  # Ensure we got a valid response

    return [
        Document(id=data["id"], page_content=data["text"], metadata=data["metadata"])
        for line in response.text.splitlines()
        if (data := json.loads(line))
    ]
