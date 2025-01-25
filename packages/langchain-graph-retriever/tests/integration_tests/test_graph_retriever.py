from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.adapters.in_memory import InMemoryListAdapter
from langchain_graph_retriever.vector_stores.in_memory import InMemoryList


def test_infers_adapter() -> None:
    # Some vector stores require at least one document to be created.
    doc = Document(
        id="doc",
        page_content="lorem ipsum and whatnot",
    )
    store = InMemoryList.from_documents([doc], FakeEmbeddings(size=8))

    retriever = GraphRetriever(
        store=store,
        edges=[],
    )

    assert isinstance(retriever.adapter, InMemoryListAdapter)
