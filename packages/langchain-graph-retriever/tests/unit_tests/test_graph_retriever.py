from langchain_core.embeddings import FakeEmbeddings
from langchain_graph_retriever import GraphRetriever
from langchain_graph_retriever.adapters.in_memory import InMemoryListAdapter
from langchain_graph_retriever.strategies import Mmr
from langchain_graph_retriever.vector_stores.in_memory import InMemoryList


def test_mmr_parameters() -> None:
    # Makes sure that copying the MMR strategy creates new embeddings.
    mmr1 = Mmr()
    mmr1._query_embedding = [0.25, 0.5, 0.75]
    assert id(mmr1._nd_query_embedding) == id(mmr1._nd_query_embedding)

    mmr2 = mmr1.model_copy(deep=True)
    assert id(mmr1._nd_query_embedding) != id(mmr2._nd_query_embedding)


def test_init_parameters_override_strategy() -> None:
    store = InMemoryListAdapter(vector_store=InMemoryList(FakeEmbeddings(size=8)))
    retriever = GraphRetriever(store=store, edges=[], k=87)  # type: ignore[call-arg]
    assert retriever.strategy.k == 87
