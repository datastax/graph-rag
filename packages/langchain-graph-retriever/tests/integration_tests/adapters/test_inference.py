from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_graph_retriever.adapters import infer_adapter, Adapter

from tests.integration_tests.stores import StoreFactory

def test_infer_store(store_factory: StoreFactory) -> None:
    # Some vector stores require at least one document to be created.
    doc = Document(
        id="doc",
        page_content="lorem ipsum and whatnot",
    )
    store = store_factory._create_store("foo", [doc], FakeEmbeddings(size=8))

    adapter = infer_adapter(store)
    assert isinstance(adapter, Adapter)