from langchain_core.documents import Document
from langchain_graph_retriever.vector_stores.in_memory import InMemoryFlat, InMemoryList
from tests.assertions import sorted_doc_ids
from tests.embeddings.simple_embeddings import AnimalEmbeddings


def test_in_memory_flat(animal_docs: list[Document]):
    vector_store = InMemoryFlat.from_documents(
        documents=animal_docs, embedding=AnimalEmbeddings()
    )

    # vector search without filter
    docs = vector_store.similarity_search("insects")
    assert sorted_doc_ids(docs) == ["butterfly", "cockroach", "grasshopper", "mosquito"]

    # vector search with with key-value filter, where value is a string
    docs = vector_store.similarity_search("insects", filter={"diet": "omnivorous"})
    assert sorted_doc_ids(docs) == ["ant", "bear", "blue jay", "cockroach"]

    # vector search with with key-value filter, where value is a list of strings
    docs = vector_store.similarity_search("insects", filter={"keywords": "pollinator"})
    assert sorted_doc_ids(docs) == []


def test_in_memory_list(animal_docs: list[Document]):
    vector_store = InMemoryList.from_documents(
        documents=animal_docs, embedding=AnimalEmbeddings()
    )

    # vector search without filter
    docs = vector_store.similarity_search("insects")
    assert sorted_doc_ids(docs) == ["butterfly", "cockroach", "grasshopper", "mosquito"]

    # vector search with with key-value filter, where value is a string
    docs = vector_store.similarity_search("insects", filter={"diet": "omnivorous"})
    assert sorted_doc_ids(docs) == ["ant", "bear", "blue jay", "cockroach"]

    # vector search with with key-value filter, where value is a list of strings
    docs = vector_store.similarity_search("insects", filter={"keywords": "pollinator"})
    assert sorted_doc_ids(docs) == ["ant", "bat", "bee", "butterfly"]
