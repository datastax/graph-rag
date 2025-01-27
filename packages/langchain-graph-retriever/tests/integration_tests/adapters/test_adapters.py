import abc
import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.base import METADATA_EMBEDDING_KEY, Adapter
from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    DENORMALIZED_KEYS_KEY,
)
from langchain_graph_retriever.types import Edge, MetadataEdge

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


@dataclass
class GetCase:
    request: list[str]
    expected: list[str]


GET_CASES: dict[str, GetCase] = {
    # Currently, this is not required for `get` implementations since the
    # traversal skips making `get` calls with no IDs. Some stores (such as chroma)
    # fail in this case.
    # "none": GetCase([], []),
    "one": GetCase(["boar"], ["boar"]),
    "many": GetCase(["boar", "chinchilla", "cobra"], ["boar", "chinchilla", "cobra"]),
    "missing": GetCase(
        ["boar", "chinchilla", "unicorn", "cobra"], ["boar", "chinchilla", "cobra"]
    ),
    "duplicate": GetCase(
        ["boar", "chinchilla", "boar", "cobra"], ["boar", "chinchilla", "cobra"]
    ),
}


@pytest.fixture(params=GET_CASES.keys())
def get_case(request) -> GetCase:
    return GET_CASES[request.param]


@dataclass
class SimilaritySearchCase:
    query: str
    expected: list[str]
    k: int | None = None
    filter: dict[str, str] | None = None

    skips: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def kwargs(self):
        kwargs = {}
        if self.k is not None:
            kwargs["k"] = self.k
        if self.filter is not None:
            kwargs["filter"] = self.filter
        return kwargs


SIMILARITY_SEARCH_CASES: dict[str, SimilaritySearchCase] = {
    "basic": SimilaritySearchCase(
        "domesticated hunters", ["cat", "horse", "chicken", "llama"]
    ),
    "k_2": SimilaritySearchCase("domesticated hunters", k=2, expected=["cat", "horse"]),
    # Many stores fail in this case. Generally it doesn't happen in the code, since
    # no IDs means we don't need to make the call. Not currently part of the contract.
    # "k_0": SimilaritySearchCase("domesticated hunters", k=0, expected=[]),
    "value_filter": SimilaritySearchCase(
        "domesticated hunters",
        filter={"type": "mammal"},
        expected=["cat", "dog", "horse", "llama"],
    ),
    "list_filter": SimilaritySearchCase(
        "domesticated hunters", filter={"keywords": "hunting"}, expected=["cat"]
    ),
    "two_filters": SimilaritySearchCase(
        "domesticated hunters",
        filter={"type": "mammal", "diet": "carnivorous"},
        expected=["cat", "dingo", "ferret"],
        skips={"chroma": "does not support multiple filters"},
    ),
    # OpenSearch supports filtering on multiple values, but it is not currently
    # relied on. Since no other adapters support it, we don't test it nor should
    # traversal depend on it.
    # "multi_list_filter": SimilaritySearchCase(
    #   "domesticated hunters",
    #   filter={"keywords": ["hunting", "agile"]},
    #   expected=["cat", "fox", "gazelle", "mongoose"]
    # ),
}


@pytest.fixture(params=SIMILARITY_SEARCH_CASES.keys())
def similarity_search_case(store_param: str, request) -> SimilaritySearchCase:
    case = SIMILARITY_SEARCH_CASES[request.param]
    skip = case.skips.get(store_param, None)
    if skip is not None:
        pytest.skip(skip)
    return case


@dataclass
class GetAdjacentCase:
    query: str
    outgoing_edges: set[Edge]
    expected: list[str]

    adjacent_k: int = 4
    filter: dict[str, Any] | None = None


GET_ADJACENT_CASES: dict[str, GetAdjacentCase] = {
    "one_edge": GetAdjacentCase(
        "domesticated hunters",
        outgoing_edges={MetadataEdge("type", "mammal")},
        expected=["horse", "llama", "dog", "cat"],
    ),
    # Note: Currently, all stores implement get adjacent by performing a
    # separate search for each edge. This means that it returns up to
    # `adjacent_k * len(outgoing_edges)` results. This will not be true if some
    # stores (eg., OpenSearch) implement get adjacent more efficiently. We may
    # wish to have `get_adjacent` select the top `adjacent_k` by sorting by
    # similarity internally to better reflect this.
    "two_edges_same_field": GetAdjacentCase(
        "domesticated hunters",
        outgoing_edges={
            MetadataEdge("type", "mammal"),
            MetadataEdge("type", "crustacean"),
        },
        expected=[
            "cat",
            "crab",
            "dog",
            "horse",
            "llama",
            "lobster",
        ],
    ),
}


@pytest.fixture(params=GET_ADJACENT_CASES.keys())
def get_adjacent_case(request) -> GetAdjacentCase:
    return GET_ADJACENT_CASES[request.param]


class AdapterComplianceSuite:
    def test_get(self, adapter: Adapter, get_case: GetCase) -> None:
        results = adapter.get(get_case.request)
        assert_ids_any_order(results, get_case.expected)

    async def test_aget(self, adapter: Adapter, get_case: GetCase) -> None:
        results = await adapter.aget(get_case.request)
        assert_ids_any_order(results, get_case.expected)

    def test_similarity_search_with_embedding(
        self, adapter: Adapter, similarity_search_case: SimilaritySearchCase
    ) -> None:
        embedding, results = adapter.similarity_search_with_embedding(
            similarity_search_case.query, **similarity_search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, similarity_search_case.expected)

    async def test_asimilarity_search_with_embedding(
        self, adapter: Adapter, similarity_search_case: SimilaritySearchCase
    ) -> None:
        embedding, results = await adapter.asimilarity_search_with_embedding(
            similarity_search_case.query, **similarity_search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, similarity_search_case.expected)

    def test_similarity_search_with_embedding_by_vector(
        self, adapter: Adapter, similarity_search_case: SimilaritySearchCase
    ) -> None:
        embedding = adapter._safe_embedding.embed_query(
            text=similarity_search_case.query
        )
        results = adapter.similarity_search_with_embedding_by_vector(
            embedding, **similarity_search_case.kwargs
        )
        assert_ids_any_order(results, similarity_search_case.expected)

    async def test_asimilarity_search_with_embedding_by_vector(
        self, adapter: Adapter, similarity_search_case: SimilaritySearchCase
    ) -> None:
        embedding = adapter._safe_embedding.embed_query(
            text=similarity_search_case.query
        )
        results = await adapter.asimilarity_search_with_embedding_by_vector(
            embedding, **similarity_search_case.kwargs
        )
        assert_ids_any_order(results, similarity_search_case.expected)

    async def test_get_adjacent(
        self, adapter: Adapter, get_adjacent_case: GetAdjacentCase
    ) -> None:
        embedding = adapter._safe_embedding.embed_query(text=get_adjacent_case.query)
        results = adapter.get_adjacent(
            outgoing_edges=get_adjacent_case.outgoing_edges,
            query_embedding=embedding,
            adjacent_k=get_adjacent_case.adjacent_k,
            filter=get_adjacent_case.filter,
        )
        assert_ids_any_order(results, get_adjacent_case.expected)

    async def test_aget_adjacent(
        self, adapter: Adapter, get_adjacent_case: GetAdjacentCase
    ) -> None:
        embedding = adapter._safe_embedding.embed_query(text=get_adjacent_case.query)
        results = await adapter.aget_adjacent(
            outgoing_edges=get_adjacent_case.outgoing_edges,
            query_embedding=embedding,
            adjacent_k=get_adjacent_case.adjacent_k,
            filter=get_adjacent_case.filter,
        )
        assert_ids_any_order(results, get_adjacent_case.expected)


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
