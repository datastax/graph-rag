from langchain_core.vectorstores import InMemoryVectorStore
from langchain_graph_retriever import GraphRetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests

from tests.animal_docs import load_animal_docs
from tests.embeddings.simple_embeddings import AnimalEmbeddings


class TestGraphTraversalRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[GraphRetriever]:
        return GraphRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        store = InMemoryVectorStore(embedding=AnimalEmbeddings())
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
