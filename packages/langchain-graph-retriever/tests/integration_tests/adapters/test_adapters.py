import abc
from collections.abc import Iterable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.base import METADATA_EMBEDDING_KEY, Adapter

from tests.animal_docs import load_animal_docs
from tests.embeddings.simple_embeddings import AnimalEmbeddings
from tests.integration_tests.stores import StoreFactory


def assert_ids_any_order(results: Iterable[Document], expected: list[str]) -> None:
    assert sorted([r.id for r in results]) == expected

def assert_valid_results(results: Iterable[Document]) -> None:
    for doc in results:
        assert doc.id is not None, "documents must have IDs"

        embedding = doc.metadata.get(METADATA_EMBEDDING_KEY)
        assert isinstance(embedding, list) and all(
            [isinstance(n, float) for n in embedding]
        ), f"metadata[METADATA_EMBEDDING_KEY] must be a list[float], but was {embedding}"


class AdapterComplianceSuite:
    def test_get_one(self, adapter: Adapter) -> None:
        results = adapter.get(["boar"])
        assert_ids_any_order(results, ["boar"])
        assert_valid_results(results)

    async def test_aget_one(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar"])
        assert_ids_any_order(results, ["boar"])
        assert_valid_results(results)

    def test_get_many(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)

    async def test_aget_many(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)

    def test_get_missing(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "unicorn", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)

    async def test_aget_missing(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "unicorn", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)

    def test_get_duplicate(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "boar", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)

    async def test_aget_duplicate(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "boar", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])
        assert_valid_results(results)


class TestBuiltinAdapters(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self, store_factory: StoreFactory, request: pytest.FixtureRequest
    ) -> Adapter:
        return store_factory.create(
            request,
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
