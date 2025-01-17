from typing import Type

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_graph_retriever import GraphTraversalRetriever
from langchain_graph_retriever.adapters.in_memory import InMemoryAdapter
from langchain_graph_retriever.strategy.eager import Eager
from langchain_tests.integration_tests import RetrieversIntegrationTests
from tests.embeddings.simple_embeddings import AnimalEmbeddings
from tests.integration_tests.animal_docs import get_animal_docs


class TestGraphTraversalRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[GraphTraversalRetriever]:
        return GraphTraversalRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        store = InMemoryVectorStore(embedding=AnimalEmbeddings())
        store.add_documents(get_animal_docs())
        adapter = InMemoryAdapter(vector_store=store)
        strategy = Eager(k=2, start_k=2, max_depth=2)
        return {
            "store": adapter,
            "edges": ["habitat"],
            "strategy": strategy,
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "what are some small agile mammals?"
