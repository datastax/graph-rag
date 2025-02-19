import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from graph_retriever import Content
from graph_retriever.adapters import Adapter
from graph_retriever.edges import Edge, IdEdge, MetadataEdge


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


@dataclass(kw_only=True)
class AdapterComplianceCase(abc.ABC):
    """
    Base dataclass for test cases.

    Attributes
    ----------
    id :
        The ID of the test case.

    expected :
        The expected results of the case.
    """

    id: str
    expected: list[str]

    requires_nested: bool = False
    requires_dict_in_list: bool = False


@dataclass
class GetCase(AdapterComplianceCase):
    """A test case for `get` and `aget`."""

    request: list[str]
    filter: dict[str, Any] | None = None


GET_CASES: list[GetCase] = [
    # Currently, this is not required for `get` implementations since the
    # traversal skips making `get` calls with no IDs. Some stores (such as chroma)
    # fail in this case.
    # GetCase("none", [], []),
    GetCase(id="one", request=["boar"], expected=["boar"]),
    GetCase(
        id="many",
        request=["boar", "chinchilla", "cobra"],
        expected=["boar", "chinchilla", "cobra"],
    ),
    GetCase(
        id="missing",
        request=["boar", "chinchilla", "unicorn", "cobra"],
        expected=["boar", "chinchilla", "cobra"],
    ),
    GetCase(
        id="duplicate",
        request=["boar", "chinchilla", "boar", "cobra"],
        expected=["boar", "chinchilla", "cobra"],
    ),
    GetCase(
        id="filtered",
        request=["boar", "chinchilla", "boar", "cobra"],
        expected=["chinchilla"],
        filter={"keywords": "andes"},
    ),
]


@dataclass
class SearchCase(AdapterComplianceCase):
    """A test case for `similarity_search_*` and `asimilarity_search_*` methods."""

    query: str
    k: int | None = None
    filter: dict[str, str] | None = None

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
    SearchCase(
        id="basic",
        query="domesticated hunters",
        expected=["cat", "horse", "chicken", "dog"],
    ),
    SearchCase(id="k2", query="domesticated hunters", k=2, expected=["cat", "horse"]),
    SearchCase(
        id="k0",
        query="domesticated hunters",
        k=0,
        expected=[],
    ),
    SearchCase(
        id="value_filter",
        query="domesticated hunters",
        filter={"type": "mammal"},
        expected=["cat", "dog", "horse", "alpaca"],
    ),
    SearchCase(
        id="list_filter",
        query="domesticated hunters",
        filter={"keywords": "hunting"},
        expected=["cat"],
    ),
    SearchCase(
        id="two_filters",
        query="domesticated hunters",
        filter={"type": "mammal", "diet": "carnivorous"},
        expected=["cat", "dingo", "ferret"],
    ),
    # OpenSearch supports filtering on multiple values, but it is not currently
    # relied on. Since no other adapters support it, we don't test it nor should
    # traversal depend on it.
    # SimilaritySearchCase(
    #   id="multi_list_filter",
    #   query="domesticated hunters",
    #   filter={"keywords": ["hunting", "agile"]},
    #   expected=["cat", "fox", "gazelle", "mongoose"]
    # ),
]


@dataclass
class AdjacentCase(AdapterComplianceCase):
    """A test case for `get_adjacent` and `aget_adjacent`."""

    query: str
    edges: set[Edge]

    k: int = 4
    filter: dict[str, Any] | None = None


ADJACENT_CASES: list[AdjacentCase] = [
    AdjacentCase(
        id="one_edge",
        query="domesticated hunters",
        edges={MetadataEdge("type", "mammal")},
        expected=["horse", "alpaca", "dog", "cat"],
    ),
    AdjacentCase(
        id="two_edges_same_field",
        query="domesticated hunters",
        edges={
            MetadataEdge("type", "mammal"),
            MetadataEdge("type", "crustacean"),
        },
        expected=[
            "alpaca",
            "cat",
            "dog",
            "horse",
        ],
    ),
    AdjacentCase(
        id="numeric",
        query="domesticated hunters",
        edges={
            MetadataEdge("number_of_legs", 0),
        },
        k=20,  # more than match the filter so we get all
        expected=[
            "barracuda",
            "cobra",
            "dolphin",
            "eel",
            "fish",
            "jellyfish",
            "manatee",
            "narwhal",
        ],
    ),
    AdjacentCase(
        id="two_edges_diff_field",
        query="domesticated hunters",
        edges={
            MetadataEdge("type", "reptile"),
            MetadataEdge("number_of_legs", 0),
        },
        k=20,  # more than match the filter so we get all
        expected=[
            "alligator",
            "barracuda",
            "chameleon",
            "cobra",
            "crocodile",
            "dolphin",
            "eel",
            "fish",
            "gecko",
            "iguana",
            "jellyfish",
            "komodo dragon",
            "lizard",
            "manatee",
            "narwhal",
        ],
    ),
    AdjacentCase(
        id="one_ids",
        query="domesticated hunters",
        edges={
            IdEdge("cat"),
        },
        expected=[
            "cat",
        ],
    ),
    AdjacentCase(
        id="many_ids",
        query="domesticated hunters",
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
        id="ids_limit_k",
        query="domesticated hunters",
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
        id="filtered_ids",
        query="domesticated hunters",
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
        id="metadata_and_id",
        query="domesticated hunters",
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
    AdjacentCase(
        id="dict_in_list",
        query="domesticated hunters",
        edges={
            MetadataEdge("tags", {"a": 5, "b": 7}),
        },
        expected=[
            "aardvark",
        ],
        requires_dict_in_list=True,
    ),
    AdjacentCase(
        id="dict_in_list_multiple",
        query="domesticated hunters",
        edges={
            MetadataEdge("tags", {"a": 5, "b": 7}),
            MetadataEdge("tags", {"a": 5, "b": 8}),
        },
        expected=[
            "aardvark",
            "albatross",
        ],
        requires_dict_in_list=True,
    ),
    AdjacentCase(
        id="absent_dict",
        query="domesticated hunters",
        edges={
            MetadataEdge("tags", {"a": 5, "b": 10}),
        },
        expected=[],
        requires_dict_in_list=True,
    ),
    AdjacentCase(
        id="nested",
        query="domesticated hunters",
        edges={
            MetadataEdge("nested.a", 5),
        },
        expected=[
            "alligator",
            "alpaca",
        ],
        requires_nested=True,
    ),
    AdjacentCase(
        id="nested_same_field",
        query="domesticated hunters",
        edges={
            MetadataEdge("nested.a", 5),
            MetadataEdge("nested.a", 6),
        },
        expected=[
            "alligator",
            "alpaca",
            "ant",
        ],
        requires_nested=True,
    ),
    AdjacentCase(
        id="nested_diff_field",
        query="domesticated hunters",
        edges={
            MetadataEdge("nested.a", 5),
            MetadataEdge("nested.b", 5),
        },
        expected=[
            "alligator",
            "alpaca",
            "anteater",
        ],
        requires_nested=True,
    ),
]


class AdapterComplianceSuite(abc.ABC):
    """
    Test suite for adapter compliance.

    To use this, create a sub-class containing a `@pytest.fixture` named
    `adapter` which returns an `Adapter` with the documents from `animals.jsonl`
    loaded.
    """

    def supports_nested_metadata(self) -> bool:
        """Return whether nested metadata is expected to work."""
        return True

    def supports_dict_in_list(self) -> bool:
        """Return whether dicts can appear in list fields in metadata."""
        return True

    def expected(self, method: str, case: AdapterComplianceCase) -> list[str]:
        """
        Override to change the expected behavior of a case.

        If the test is expected to fail, call `pytest.xfail(reason)`, or
        `pytest.skip(reason)` if it can't be executed.

        Generally, this should *not* change the expected results, unless the the
        adapter being tested uses wildly different distance metrics or a
        different embedding. The `AnimalsEmbedding` is deterimistic and the
        results across vector stores should generally be deterministic and
        consistent.

        Parameters
        ----------
        method :
            The method being tested. For instance, `get`, `aget`, or
            `similarity_search_with_embedding`, etc.
        case :
            The case being tested.

        Returns
        -------
        :
            The expected animals.
        """
        if not self.supports_nested_metadata() and case.requires_nested:
            pytest.xfail("nested metadata not supported")
        if not self.supports_dict_in_list() and case.requires_dict_in_list:
            pytest.xfail("dict-in-list fields is not supported")
        return case.expected

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
        expected = self.expected("get", get_case)
        results = adapter.get(get_case.request, filter=get_case.filter)
        assert_ids_any_order(results, expected)

    async def test_aget(self, adapter: Adapter, get_case: GetCase) -> None:
        """Run tests for `aget`."""
        expected = self.expected("aget", get_case)
        results = await adapter.aget(get_case.request, filter=get_case.filter)
        assert_ids_any_order(results, expected)

    def test_search_with_embedding(
        self, adapter: Adapter, search_case: SearchCase
    ) -> None:
        """Run tests for `search_with_embedding`."""
        expected = self.expected("search_with_embedding", search_case)
        embedding, results = adapter.search_with_embedding(
            search_case.query, **search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, expected)

    async def test_asearch_with_embedding(
        self, adapter: Adapter, search_case: SearchCase
    ) -> None:
        """Run tests for `asearch_with_embedding`."""
        expected = self.expected("asearch_with_embedding", search_case)
        embedding, results = await adapter.asearch_with_embedding(
            search_case.query, **search_case.kwargs
        )
        assert_is_embedding(embedding)
        assert_ids_any_order(results, expected)

    def test_search(self, adapter: Adapter, search_case: SearchCase) -> None:
        """Run tests for `search`."""
        expected = self.expected("search", search_case)
        embedding, _ = adapter.search_with_embedding(search_case.query, k=0)
        results = adapter.search(embedding, **search_case.kwargs)
        assert_ids_any_order(results, expected)

    async def test_asearch(self, adapter: Adapter, search_case: SearchCase) -> None:
        """Run tests for `asearch`."""
        expected = self.expected("asearch", search_case)
        embedding, _ = await adapter.asearch_with_embedding(search_case.query, k=0)
        results = await adapter.asearch(embedding, **search_case.kwargs)
        assert_ids_any_order(results, expected)

    def test_adjacent(self, adapter: Adapter, adjacent_case: AdjacentCase) -> None:
        """Run tests for `adjacent."""
        expected = self.expected("adjacent", adjacent_case)
        embedding, _ = adapter.search_with_embedding(adjacent_case.query, k=0)
        results = adapter.adjacent(
            edges=adjacent_case.edges,
            query_embedding=embedding,
            k=adjacent_case.k,
            filter=adjacent_case.filter,
        )
        assert_ids_any_order(results, expected)

    async def test_aadjacent(
        self, adapter: Adapter, adjacent_case: AdjacentCase
    ) -> None:
        """Run tests for `aadjacent."""
        expected = self.expected("aadjacent", adjacent_case)
        embedding, _ = await adapter.asearch_with_embedding(adjacent_case.query, k=0)
        results = await adapter.aadjacent(
            edges=adjacent_case.edges,
            query_embedding=embedding,
            k=adjacent_case.k,
            filter=adjacent_case.filter,
        )
        assert_ids_any_order(results, expected)
