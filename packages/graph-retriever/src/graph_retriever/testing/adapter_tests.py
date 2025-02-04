import abc
import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from graph_retriever import Adapter, Edge, IdEdge, MetadataEdge
from graph_retriever.content import Content


def assert_valid_result(content: Content):
    """Assert the content is valid."""
    assert isinstance(content.id, str)
    assert_is_embedding(content.embedding)


def assert_is_embedding(value: Any):
    """Assert the value is an embedding."""
    assert isinstance(value, list)
    for item in value:
        assert isinstance(item, float)


def assert_valid_results(docs: Iterable[Content]):
    """Assert all of the contents are valid results."""
    for doc in docs:
        assert_valid_result(doc)


def assert_ids_any_order(
    results: Iterable[Content],
    expected: list[str],
) -> None:
    """Assert the results are valid and match the IDs."""
    assert_valid_results(results)

    result_ids = [r.id for r in results]
    assert set(result_ids) == set(expected), "should contain exactly expected IDs"


@dataclass
class GetCase:
    """A test case for `get` and `aget`."""

    id: str
    request: list[str]
    expected: list[str]
    filter: dict[str, Any] | None = None


GET_CASES: list[GetCase] = [
    # Currently, this is not required for `get` implementations since the
    # traversal skips making `get` calls with no IDs. Some stores (such as chroma)
    # fail in this case.
    # GetCase("none", [], []),
    GetCase("one", ["boar"], ["boar"]),
    GetCase("many", ["boar", "chinchilla", "cobra"], ["boar", "chinchilla", "cobra"]),
    GetCase(
        "missing",
        ["boar", "chinchilla", "unicorn", "cobra"],
        ["boar", "chinchilla", "cobra"],
    ),
    GetCase(
        "duplicate",
        ["boar", "chinchilla", "boar", "cobra"],
        ["boar", "chinchilla", "cobra"],
    ),
    GetCase(
        "filtered",
        ["boar", "chinchilla", "boar", "cobra"],
        ["chinchilla"],
        filter={"keywords": "andes"},
    ),
]


@dataclass
class SearchCase:
    """A test case for `similarity_search_*` and `asimilarity_search_*` methods."""

    id: str
    query: str
    expected: list[str]
    k: int | None = None
    filter: dict[str, str] | None = None

    skips: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def kwargs(self):
        """Return keyword arguments for the test invocation."""
        kwargs = {}
        if self.k is not None:
            kwargs["k"] = self.k
        if self.filter is not None:
            kwargs["filter"] = self.filter
        return kwargs


SEARCH_CASES: list[SearchCase] = [
    SearchCase("basic", "domesticated hunters", ["cat", "horse", "chicken", "llama"]),
    SearchCase("k2", "domesticated hunters", k=2, expected=["cat", "horse"]),
    # Many stores fail in this case. Generally it doesn't happen in the code, since
    # no IDs means we don't need to make the call. Not currently part of the contract.
    # SimilaritySearchCase("k0", "domesticated hunters", k=0, expected=[]),
    SearchCase(
        "value_filter",
        "domesticated hunters",
        filter={"type": "mammal"},
        expected=["cat", "dog", "horse", "llama"],
    ),
    SearchCase(
        "list_filter",
        "domesticated hunters",
        filter={"keywords": "hunting"},
        expected=["cat"],
    ),
    SearchCase(
        "two_filters",
        "domesticated hunters",
        filter={"type": "mammal", "diet": "carnivorous"},
        expected=["cat", "dingo", "ferret"],
    ),
    # OpenSearch supports filtering on multiple values, but it is not currently
    # relied on. Since no other adapters support it, we don't test it nor should
    # traversal depend on it.
    # SimilaritySearchCase(
    #   "multi_list_filter",
    #   "domesticated hunters",
    #   filter={"keywords": ["hunting", "agile"]},
    #   expected=["cat", "fox", "gazelle", "mongoose"]
    # ),
]


@dataclass
class AdjacentCase:
    """A test case for `get_adjacent` and `aget_adjacent`."""

    id: str
    query: str
    edges: set[Edge]
    expected: list[str]

    k: int = 4
    filter: dict[str, Any] | None = None


ADJACENT_CASES: list[AdjacentCase] = [
    AdjacentCase(
        "one_edge",
        "domesticated hunters",
        edges={MetadataEdge("type", "mammal")},
        expected=["horse", "llama", "dog", "cat"],
    ),
    AdjacentCase(
        "two_edges_same_field",
        "domesticated hunters",
        edges={
            MetadataEdge("type", "mammal"),
            MetadataEdge("type", "crustacean"),
        },
        expected=[
            "cat",
            "dog",
            "horse",
            "llama",
        ],
    ),
    AdjacentCase(
        "one_ids",
        "domesticated hunters",
        edges={
            IdEdge("cat"),
        },
        expected=[
            "cat",
        ],
    ),
    AdjacentCase(
        "many_ids",
        "domesticated hunters",
        edges={
            IdEdge("cat"),
            IdEdge("dog"),
            IdEdge("unicorn"),
            IdEdge("crab"),
        },
        expected=[
            "cat",
            "dog",
            "crab",
        ],
    ),
    AdjacentCase(
        "ids_limit_k",
        "domesticated hunters",
        edges={
            IdEdge("cat"),
            IdEdge("dog"),
            IdEdge("unicorn"),
            IdEdge("antelope"),
        },
        k=2,
        expected=[
            "cat",
            "dog",
        ],
    ),
    AdjacentCase(
        "filtered_ids",
        "domesticated hunters",
        edges={
            IdEdge("boar"),
            IdEdge("chinchilla"),
            IdEdge("unicorn"),
            IdEdge("griaffe"),
        },
        filter={"keywords": "andes"},
        expected=[
            "chinchilla",
        ],
    ),
    AdjacentCase(
        "metadata_and_id",
        "domesticated hunters",
        edges={
            IdEdge("cat"),
            MetadataEdge("type", "reptile"),
        },
        k=6,
        expected=[
            "alligator",  # reptile
            "crocodile",  # reptile
            "cat",  # by ID
            "chameleon",  # reptile
            "gecko",  # reptile
            "komodo dragon",  # reptile
        ],
    ),
]


class AdapterComplianceSuite(abc.ABC):
    """
    Test suite for adapter compliance.

    To use this, create a sub-class containing a `@pytest.fixture` named
    `adapter` which returns an `Adapter` with the documents from `animals.jsonl`
    loaded.
    """

    def skip_case(self, method: str, case_id: str) -> None:
        """
        Override to skip a specific method or test case.

        Call `pytest.skip(reason)` if necessary with the reason to skip,
        or `pytest.xfail(reason)` if a failure is expected.

        Parameters
        ----------
        method : str
            The method being tested. For instance, `get`, `aget`, or
            `similarity_search_with_embedding`, etc.
        case_id : str
            The ID of the case being executed. For instance, `one` or `many`.
        """

    @pytest.fixture(params=GET_CASES, ids=lambda c: c.id)
    def get_case(self, request) -> GetCase:
        """Fixture providing the `get` and `aget` test cases."""
        return request.param

    @pytest.fixture(params=ADJACENT_CASES, ids=lambda c: c.id)
    def adjacent_case(self, request) -> AdjacentCase:
        """Fixture providing the `get_adjacent` and `aget_adjacent` test cases."""
        return request.param

    @pytest.fixture(params=SEARCH_CASES, ids=lambda c: c.id)
    def search_case(self, request) -> SearchCase:
        """Fixture providing the `(a)?similarity_search_*` test cases."""
        return request.param

    def test_get(self, adapter: Adapter, get_case: GetCase) -> None:
        """Run tests for `get`."""
        self.skip_case("get", get_case.id)
        results = adapter.get(get_case.request, filter=get_case.filter)
        assert_ids_any_order(results, get_case.expected)

    async def test_aget(self, adapter: Adapter, get_case: GetCase) -> None:
        """Run tests for `aget`."""
        self.skip_case("aget", get_case.id)
        results = await adapter.aget(get_case.request, filter=get_case.filter)
        assert_ids_any_order(results, get_case.expected)

    def test_search_with_embedding(
        self, adapter: Adapter, search_case: SearchCase
    ) -> None:
        """Run tests for `search_with_embedding`."""
        self.skip_case("search_with_embedding", search_case.id)
        embedding, results = adapter.search_with_embedding(
            search_case.query, **search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, search_case.expected)

    async def test_asearch_with_embedding(
        self, adapter: Adapter, search_case: SearchCase
    ) -> None:
        """Run tests for `asearch_with_embedding`."""
        self.skip_case("asearch_with_embedding", search_case.id)
        embedding, results = await adapter.asearch_with_embedding(
            search_case.query, **search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, search_case.expected)

    def test_search(self, adapter: Adapter, search_case: SearchCase) -> None:
        """Run tests for `search`."""
        self.skip_case("search", search_case.id)
        embedding = adapter.embed_query(search_case.query)
        results = adapter.search(embedding, **search_case.kwargs)
        assert_ids_any_order(results, search_case.expected)

    async def test_asearch(self, adapter: Adapter, search_case: SearchCase) -> None:
        """Run tests for `asearch`."""
        self.skip_case("asearch", search_case.id)
        embedding = adapter.embed_query(search_case.query)
        results = await adapter.asearch(embedding, **search_case.kwargs)
        assert_ids_any_order(results, search_case.expected)

    async def test_adjacent(
        self, adapter: Adapter, adjacent_case: AdjacentCase
    ) -> None:
        """Run tests for `adjacent."""
        self.skip_case("adjacent", adjacent_case.id)
        embedding = adapter.embed_query(adjacent_case.query)
        results = adapter.adjacent(
            edges=adjacent_case.edges,
            query_embedding=embedding,
            k=adjacent_case.k,
            filter=adjacent_case.filter,
        )
        assert_ids_any_order(results, adjacent_case.expected)

    async def test_aadjacent(
        self, adapter: Adapter, adjacent_case: AdjacentCase
    ) -> None:
        """Run tests for `aadjacent."""
        self.skip_case("aadjacent", adjacent_case.id)
        embedding = adapter.embed_query(adjacent_case.query)
        results = await adapter.aadjacent(
            edges=adjacent_case.edges,
            query_embedding=embedding,
            k=adjacent_case.k,
            filter=adjacent_case.filter,
        )
        assert_ids_any_order(results, adjacent_case.expected)
