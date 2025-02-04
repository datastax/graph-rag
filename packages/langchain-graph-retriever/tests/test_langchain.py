from graph_retriever.testing.embeddings import AnimalEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_graph_retriever import GraphRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from tests.conftest import load_animal_docs
from tests.embeddings import BaseEmbeddings


class TestGraphTraversalRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[GraphRetriever]:
        return GraphRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        embedding = BaseEmbeddings(AnimalEmbeddings())
        store = InMemoryVectorStore(embedding=embedding)
        store.add_documents(load_animal_docs())
        return {
            "store": store,
            "edges": [("habitat", "habitat")],
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "what are some small agile mammals?"
