from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.adapters.in_memory import InMemoryAdapter


def test_infers_adapter() -> None:
    # Some vector stores require at least one document to be created.
    doc = Document(
        id="doc",
        page_content="lorem ipsum and whatnot",
    )
    store = InMemoryVectorStore.from_documents([doc], FakeEmbeddings(size=8))

    retriever = GraphRetriever(
        store=store,
        edges=[],
    )

    assert isinstance(retriever.adapter, InMemoryAdapter)
