from typing import Iterable

from langchain_core.documents import Document

def doc_ids(docs: Iterable[Document]) -> list[str]:
    return [doc.id for doc in docs if doc.id is not None]


def sorted_doc_ids(docs: Iterable[Document]) -> list[str]:
    return sorted(doc_ids(docs=docs))


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata
