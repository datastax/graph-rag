import abc
from typing import Iterable
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from langchain_graph_retriever.adapters.base import Adapter, METADATA_EMBEDDING_KEY
from tests.integration_tests.stores import StoreFactory
from tests.embeddings.simple_embeddings import AnimalEmbeddings
from tests.animal_docs import load_animal_docs

def assert_valid_results(results: Iterable[Document]) -> None:
    for doc in results:
        assert doc.id is not None, "documents must have IDs"

        embedding = doc.metadata.get(METADATA_EMBEDDING_KEY)
        assert isinstance(embedding, list) and all([isinstance(n, float) for n in embedding]), (
            "metadata[METADATA_EMBEDDING_KEY] must be a list[float]"
        )

class AdapterComplianceSuite:
    def test_get_one(self, adapter: Adapter) -> None:
        results = adapter.get(["boar"])
        assert {r.id for r in results} == {"boar"}
        assert_valid_results(results)

    async def test_aget_one(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar"])
        assert {r.id for r in results} == {"boar"}
        assert_valid_results(results)

    def test_get_many(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)

    async def test_aget_many(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)

    def test_get_missing(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "unicorn", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)

    async def test_aget_missing(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "unicorn", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)

    def test_get_duplicate(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "boar", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)

    async def test_aget_duplicate(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "boar", "cobra"])
        assert {r.id for r in results} == {"boar", "chinchilla", "cobra"}
        assert_valid_results(results)


class TestBuiltinAdapters(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(self,
                store_factory: StoreFactory,
                request: pytest.FixtureRequest) -> Adapter:
        return store_factory.create(request,
                                    embedding=AnimalEmbeddings(),
                                    docs=load_animal_docs(),
        )

class TestAdapterCompliance(abc.ABC, AdapterComplianceSuite):
    """
    Run the AdapterComplianceSuite on a the adapter created by `make`.

    To use this, instantiate it in your `pytest` code and implement `make` to create.
    """

    @abc.abstractmethod
    def make(self, embedding: Embeddings, docs: list[Document]) -> Adapter: ...

    @pytest.fixture(scope="class")
    def adapter(self) -> Adapter:
        return self.make(embedding=AnimalEmbeddings(), docs=load_animal_docs())
