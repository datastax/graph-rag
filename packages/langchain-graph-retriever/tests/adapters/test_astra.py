import dataclasses
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from graph_retriever.testing.adapter_tests import (
    AdapterComplianceCase,
    AdapterComplianceSuite,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.astra import AstraAdapter, _metadata_queries
from typing_extensions import override


def create_metadata_queries(
    user_filters: dict[str, Any],
    metadata: dict[str, Iterable[Any]] = {},
) -> list[dict[str, Any]]:
    return list(
        _metadata_queries(
            user_filters=user_filters,
            metadata=metadata,
        )
    )


def create_metadata_query(
    user_filters: dict[str, Any],
    metadata: dict[str, Iterable[Any]] = {},
) -> dict[str, Any]:
    queries = create_metadata_queries(user_filters=user_filters, metadata=metadata)
    assert len(queries) == 1
    return queries[0]


def test_create_metadata_query_no_user() -> None:
    assert create_metadata_queries({}, metadata={}) == []

    assert create_metadata_query({}, metadata={"foo": [5]}) == {"foo": 5}

    assert create_metadata_query({}, metadata={"foo": [5, 6]}) == {
        "foo": {"$in": [5, 6]}
    }

    assert create_metadata_queries({}, metadata={"foo": [5], "bar": [7]}) == [
        {"foo": 5},
        {"bar": 7},
    ]

    assert create_metadata_queries(
        {},
        metadata={"foo": [5, 6], "bar": [7, 8]},
    ) == [
        {"foo": {"$in": [5, 6]}},
        {"bar": {"$in": [7, 8]}},
    ]

    assert create_metadata_queries(
        {}, metadata={"foo": list(range(0, 200)), "bar": [7]}
    ) == [
        {"foo": {"$in": list(range(0, 100))}},
        {"foo": {"$in": list(range(100, 200))}},
        {"bar": 7},
    ]


def test_create_metadata_query_user() -> None:
    USER = {"answer": 42}
    assert create_metadata_queries(USER, metadata={}) == []
    assert create_metadata_queries(USER, metadata={"foo": []}) == []
    assert create_metadata_query(USER, metadata={"foo": [5]}) == {
        "$and": [
            {"foo": 5},
            {"answer": 42},
        ],
    }

    assert create_metadata_query(USER, metadata={"foo": [5, 6]}) == {
        "$and": [
            {"foo": {"$in": [5, 6]}},
            {"answer": 42},
        ],
    }

    assert create_metadata_queries(USER, metadata={"foo": [5], "bar": [7]}) == [
        {
            "$and": [
                {"foo": 5},
                {"answer": 42},
            ],
        },
        {
            "$and": [
                {"bar": 7},
                {"answer": 42},
            ],
        },
    ]

    assert create_metadata_queries(
        USER,
        metadata={"foo": [5, 6], "bar": [7, 8]},
    ) == [
        {
            "$and": [
                {"foo": {"$in": [5, 6]}},
                {"answer": 42},
            ],
        },
        {
            "$and": [
                {"bar": {"$in": [7, 8]}},
                {"answer": 42},
            ],
        },
    ]

    assert create_metadata_queries(
        USER, metadata={"foo": list(range(0, 200)), "bar": [7, 8]}
    ) == [
        {
            "$and": [
                {"foo": {"$in": list(range(0, 100))}},
                {"answer": 42},
            ]
        },
        {
            "$and": [
                {"foo": {"$in": list(range(100, 200))}},
                {"answer": 42},
            ]
        },
        {
            "$and": [
                {"bar": {"$in": [7, 8]}},
                {"answer": 42},
            ]
        },
    ]


@dataclasses.dataclass
class _AstraConfig:
    token: str
    keyspace: str
    api_endpoint: str


@pytest.fixture(scope="module")
def astra_config(enabled_stores: set[str]) -> Iterator[_AstraConfig | None]:
    if "astra" not in enabled_stores:
        pytest.skip("Pass --stores=astra to test Astra")
        return

    from astrapy import DataAPIClient
    from dotenv import load_dotenv

    load_dotenv()

    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    my_client = DataAPIClient(token=token)
    admin = my_client.get_admin().get_database_admin(api_endpoint)
    admin.create_keyspace(keyspace)

    # Sometimes the creation of the store fails because the keyspace isn't
    # created yet. To avoid that, poll the list of keyspaces until we
    # confirm it is created.
    found = False
    t_end = time.time() + 5  # run 5 seconds
    while time.time() < t_end:
        keyspaces = admin.list_keyspaces()
        if keyspace in keyspaces:
            found = True
            break

        print(f"Waiting for keyspace '{keyspace}'...")  # noqa: T201
        time.sleep(0.01)

    assert found, f"Keyspace '{keyspace}' not created"
    yield _AstraConfig(token=token, keyspace=keyspace, api_endpoint=api_endpoint)

    if keyspace != "default_keyspace":
        admin.drop_keyspace(keyspace)


class TestAstraAdapter(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self,
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
        astra_config: _AstraConfig,
    ) -> Iterator["AstraAdapter"]:
        from langchain_astradb import AstraDBVectorStore

        store = AstraDBVectorStore(
            embedding=animal_embeddings,
            collection_name="animals",
            namespace=astra_config.keyspace,
            token=astra_config.token,
            api_endpoint=astra_config.api_endpoint,
            pre_delete_collection=True,
        )
        store.add_documents(animal_docs)

        yield AstraAdapter(store)

        store.delete_collection()


VECTORIZE_EXPECTATION_OVERRIDES: dict[tuple[str, str], list[str]] = {
    ("search", "basic"): ["alpaca", "cat", "chicken", "horse"],
    ("asearch", "basic"): ["alpaca", "cat", "chicken", "horse"],
    ("search_with_embedding", "basic"): ["alpaca", "cat", "chicken", "horse"],
    ("asearch_with_embedding", "basic"): ["alpaca", "cat", "chicken", "horse"],
    ("adjacent", "metadata_and_id"): [
        "cat",
        "cobra",
        "crocodile",
        "gecko",
        "iguana",
        "lizard",
    ],
    ("aadjacent", "metadata_and_id"): [
        "cat",
        "cobra",
        "crocodile",
        "gecko",
        "iguana",
        "lizard",
    ],
}


class TestAstraVectorizeAdapter(AdapterComplianceSuite):
    @override
    def expected(self, method: str, case: AdapterComplianceCase) -> list[str]:
        # Since vectorize currently requires a server-side embedding model, we
        # need to change the expectations a little to reflect the embeddings
        # that are actually computed.
        return VECTORIZE_EXPECTATION_OVERRIDES.get((method, case.id), []) or (
            super().expected(method, case)
        )

    @pytest.fixture(scope="class")
    def adapter(
        self,
        animal_docs: list[Document],
        astra_config: _AstraConfig,
    ) -> Iterator["AstraAdapter"]:
        from astrapy.info import VectorServiceOptions
        from langchain_astradb import AstraDBVectorStore

        service_options = VectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        )

        store = AstraDBVectorStore(
            collection_name="animals_vectorize",
            collection_vector_service_options=service_options,
            namespace=astra_config.keyspace,
            token=astra_config.token,
            api_endpoint=astra_config.api_endpoint,
            pre_delete_collection=True,
        )
        store.add_documents(animal_docs)

        yield AstraAdapter(store)

        store.delete_collection()
