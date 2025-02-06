import dataclasses
import os
import time
from collections.abc import Iterator

import pytest
from astrapy.authentication import StaticTokenProvider
from graph_retriever.testing.adapter_tests import (
    AdapterComplianceCase,
    AdapterComplianceSuite,
)
from langchain_astradb.utils.vector_store_codecs import _DefaultVSDocumentCodec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.astra import AstraAdapter, _QueryHelper
from typing_extensions import override

TEST_CODEC = _DefaultVSDocumentCodec("page_content", ignore_invalid_documents=True)


def test_create_ids_query_no_user() -> None:
    query_helper = _QueryHelper(TEST_CODEC, {})

    assert query_helper.create_ids_query(["1"]) == {"_id": "1"}

    query = query_helper.create_ids_query(["1", "2", "3"])
    assert query is not None
    assert query["_id"]["$in"] == ["1", "2", "3"]


def test_create_ids_query_user() -> None:
    query_helper = _QueryHelper(TEST_CODEC, {"answer": 42})

    assert query_helper.create_ids_query(["1"]) == {
        "$and": [
            {"_id": "1"},
            {"metadata.answer": 42},
        ]
    }

    query = query_helper.create_ids_query(["1", "2", "3"])
    assert query is not None
    assert query["$and"][0]["_id"]["$in"] == ["1", "2", "3"]


def test_create_metadata_query_no_user() -> None:
    query_helper = _QueryHelper(TEST_CODEC, {})

    assert query_helper.create_metadata_query({}) is None
    assert query_helper.create_metadata_query({"foo": []}) is None
    assert query_helper.create_metadata_query({"foo": [5]}) == {"metadata.foo": 5}

    query = query_helper.create_metadata_query({"foo": [5, 6]})
    assert query is not None
    assert sorted(query["metadata.foo"]["$in"]) == [5, 6]

    assert query_helper.create_metadata_query({"foo": [5], "bar": [7]}) == {
        "$or": [
            {"metadata.foo": 5},
            {"metadata.bar": 7},
        ],
    }

    query = query_helper.create_metadata_query({"foo": [5, 6], "bar": [7, 8]})
    assert query is not None
    assert sorted(query["$or"][0]["metadata.foo"]["$in"]) == [5, 6]
    assert sorted(query["$or"][1]["metadata.bar"]["$in"]) == [7, 8]

    query = query_helper.create_metadata_query({"foo": list(range(0, 200))})
    assert query is not None
    assert sorted(query["$or"][0]["metadata.foo"]["$in"]) == list(range(0, 100))
    assert sorted(query["$or"][1]["metadata.foo"]["$in"]) == list(range(100, 200))


def test_create_metadata_query_user() -> None:
    query_helper = _QueryHelper(TEST_CODEC, {"answer": 42})

    assert query_helper.create_metadata_query({}) is None
    assert query_helper.create_metadata_query({"foo": []}) is None
    assert query_helper.create_metadata_query({"foo": [5]}) == {
        "$and": [
            {"metadata.foo": 5},
            {"metadata.answer": 42},
        ],
    }

    query = query_helper.create_metadata_query({"foo": [5, 6]})
    assert query is not None
    assert query["$and"][0]["metadata.foo"]["$in"] == [5, 6]
    assert query["$and"][1] == {"metadata.answer": 42}

    assert query_helper.create_metadata_query({"foo": [5], "bar": [7]}) == {
        "$and": [
            {
                "$or": [
                    {"metadata.foo": 5},
                    {"metadata.bar": 7},
                ],
            },
            {"metadata.answer": 42},
        ],
    }

    query = query_helper.create_metadata_query({"foo": [5, 6], "bar": [7, 8]})
    assert query is not None
    assert query["$and"][0]["$or"][0]["metadata.foo"]["$in"] == [5, 6]
    assert query["$and"][0]["$or"][1]["metadata.bar"]["$in"] == [7, 8]
    assert query["$and"][1] == {"metadata.answer": 42}

    query = query_helper.create_metadata_query({"foo": list(range(0, 200))})
    assert query is not None
    assert query["$and"][0]["$or"][0]["metadata.foo"]["$in"] == list(range(0, 100))
    assert query["$and"][0]["$or"][1]["metadata.foo"]["$in"] == list(range(100, 200))
    assert query["$and"][1] == {"metadata.answer": 42}


@dataclasses.dataclass
class _AstraConfig:
    token: StaticTokenProvider
    keyspace: str
    api_endpoint: str


@pytest.fixture(scope="module")
def astra_config(enabled_stores: set[str]) -> Iterator[_AstraConfig | None]:
    if "astra" not in enabled_stores:
        pytest.skip("Pass --stores=astra to test Astra")
        return

    from astrapy import AstraDBDatabaseAdmin
    from dotenv import load_dotenv

    load_dotenv()

    token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
    keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    admin = AstraDBDatabaseAdmin(api_endpoint=api_endpoint, token=token)
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
        from astrapy.info import CollectionVectorServiceOptions
        from langchain_astradb import AstraDBVectorStore

        service = CollectionVectorServiceOptions(
            provider="nvidia",
            model_name="NV-Embed-QA",
        )

        store = AstraDBVectorStore(
            collection_name="animals_vectorize",
            collection_vector_service_options=service,
            namespace=astra_config.keyspace,
            token=astra_config.token,
            api_endpoint=astra_config.api_endpoint,
            pre_delete_collection=True,
        )
        store.add_documents(animal_docs)

        yield AstraAdapter(store)

        store.delete_collection()
