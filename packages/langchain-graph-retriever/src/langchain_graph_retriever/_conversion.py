from graph_retriever.content import Content
from graph_retriever.types import Node
from langchain_core.documents import Document

METADATA_EMBEDDING_KEY = "__embedding"


def node_to_doc(node: Node) -> Document:
    return Document(
        id=node.id,
        page_content=node.content,
        metadata={**node.extra_metadata, **node.metadata},
    )


def doc_to_content(doc: Document, *, embedding: list[float] | None = None) -> Content:
    """
    Convert a LangChain document to a `Content`.

    Parameters
    ----------
    doc :
        The document to convert.

    embedding :
        The embedding of the document. If not provided, the
        `doc.metadata[METADATA_EMBEDDING_KEY]` should be set to the embedding.

    Returns
    -------
    :
        The converted content.
    """
    assert doc.id is not None

    if embedding is None:
        embedding = doc.metadata.pop(METADATA_EMBEDDING_KEY)
    assert embedding is not None

    return Content(
        id=doc.id,
        content=doc.page_content,
        embedding=embedding,
        metadata=doc.metadata,
    )
