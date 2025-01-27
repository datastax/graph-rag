from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.adapters.in_memory import InMemoryAdapter
from langchain_graph_retriever.strategies import Mmr


def test_mmr_parameters() -> None:
    # Makes sure that copying the MMR strategy creates new embeddings.
    mmr1 = Mmr()
    mmr1._query_embedding = [0.25, 0.5, 0.75]
    assert id(mmr1._nd_query_embedding) == id(mmr1._nd_query_embedding)

    mmr2 = mmr1.model_copy(deep=True)
    assert id(mmr1._nd_query_embedding) != id(mmr2._nd_query_embedding)


def test_init_parameters_override_strategy() -> None:
    store = InMemoryAdapter(vector_store=InMemoryVectorStore(FakeEmbeddings(size=8)))
    retriever = GraphRetriever(store=store, edges=[], k=87)  # type: ignore[call-arg]
    assert retriever.strategy.k == 87
