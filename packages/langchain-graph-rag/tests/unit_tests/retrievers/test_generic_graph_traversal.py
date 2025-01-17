from langchain_graph_rag.retrievers.strategy.mmr import Mmr


def test_mmr_parameters() -> None:
    # Makes sure that copying the MMR startegy creates new embeddings.
    mmr1 = Mmr(query_embedding=[0.25, 0.5, 0.75])
    assert id(mmr1._nd_query_embedding) == id(mmr1._nd_query_embedding)

    mmr2 = mmr1.model_copy(deep=True)
    assert id(mmr1._nd_query_embedding) != id(mmr2._nd_query_embedding)
