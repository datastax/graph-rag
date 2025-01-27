import abc
from collections.abc import Iterable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.base import METADATA_EMBEDDING_KEY, Adapter
from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    DENORMALIZED_KEYS_KEY,
)

from tests.animal_docs import load_animal_docs
from tests.embeddings.simple_embeddings import AnimalEmbeddings
from tests.integration_tests.stores import AdapterFactory


def assert_valid_result(doc: Document):
    assert isinstance(doc.id, str)

    assert DENORMALIZED_KEYS_KEY not in doc.metadata
    assert METADATA_EMBEDDING_KEY in doc.metadata
    assert_is_embedding(doc.metadata[METADATA_EMBEDDING_KEY])


def assert_is_embedding(value):
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, float)


def assert_valid_results(docs: Iterable[Document]):
    for doc in docs:
        assert_valid_result(doc)


def assert_ids_any_order(results: Iterable[Document], expected: list[str]) -> None:
    assert_valid_results(results)

    result_ids = [r.id for r in results]
    assert len(set(result_ids)) == len(result_ids), "should not contain duplicates"
    assert set(result_ids) == set(expected), "should contain exactly expected IDs"


class AdapterComplianceSuite:
    def test_get_one(self, adapter: Adapter) -> None:
        results = adapter.get(["boar"])
        assert_ids_any_order(results, ["boar"])

    async def test_aget_one(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar"])
        assert_ids_any_order(results, ["boar"])

    def test_get_many(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    async def test_aget_many(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    def test_get_missing(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "unicorn", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    async def test_aget_missing(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "unicorn", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    def test_get_duplicate(self, adapter: Adapter) -> None:
        results = adapter.get(["boar", "chinchilla", "boar", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    async def test_aget_duplicate(self, adapter: Adapter) -> None:
        results = await adapter.aget(["boar", "chinchilla", "boar", "cobra"])
        assert_ids_any_order(results, ["boar", "chinchilla", "cobra"])

    def test_similarity_search_with_embedding(self, adapter: Adapter) -> None:
        embedding, results = adapter.similarity_search_with_embedding(
            "domesticated hunters"
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, ["cat", "horse", "chicken", "llama"])

    def test_similarity_search_with_embedding_respects_k(
        self, adapter: Adapter
    ) -> None:
        embedding, results = adapter.similarity_search_with_embedding(
            "domesticated hunters", k=2
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, ["cat", "horse"])

    def test_similarity_search_with_embedding_respects_value_filters(
        self, adapter: Adapter
    ) -> None:
        embedding, results = adapter.similarity_search_with_embedding(
            "domesticated hunters", filter={"type": "mammal"}
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, ["cat", "dog", "horse", "llama"])

    def test_similarity_search_with_embedding_respects_list_filters(
        self, adapter: Adapter
    ) -> None:
        embedding, results = adapter.similarity_search_with_embedding(
            "domesticated hunters", filter={"keywords": "hunting"}
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, ["cat"])

        # TODO: Add support for nested list filters
        # embedding, results = adapter.similarity_search_with_embedding(
        #     "domesticated hunters", filter={"keywords": ["hunter", "agile"]}
        # )
        # assert_is_embedding(embedding)
        # assert_ids_any_order(results, ["cat", "dog", "horse", "llama"])


class TestBuiltinAdapters(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self, adapter_factory: AdapterFactory, request: pytest.FixtureRequest
    ) -> Adapter:
        return adapter_factory.create(
            request,
            embedding=AnimalEmbeddings(),
            docs=load_animal_docs(),
            nested_metadata_fields={"keywords"},
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
