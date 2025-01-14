from typing import Iterable

import pytest
from animal_docs import animal_docs, animal_store
from invoker import invoker
from langchain_core.documents import Document
from parser_docs import graph_vector_store_docs, parser_store

# Imports for definitions.
from stores import (
    enabled_stores,
    store_factory,
    store_param,
    support_normalized_metadata,
)

# Mark these imports as used so they don't removed.
# They need to be imported here so the fixtures are available.
_ = (
    store_factory,
    store_param,
    enabled_stores,
    support_normalized_metadata,
    animal_docs,
    animal_store,
    graph_vector_store_docs,
    parser_store,
    invoker,
)


def sorted_doc_ids(docs: Iterable[Document]) -> list[str]:
    return sorted([doc.id for doc in docs if doc.id is not None])


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


@pytest.fixture(scope="module")
def hello_docs() -> list[Document]:
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


@pytest.fixture(scope="module")
def mmr_docs() -> list[Document]:
    """The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

            ______ v2
          //     \\
         //        \\  v1
    v3   |     .    | query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    return [v0, v1, v2, v3]
