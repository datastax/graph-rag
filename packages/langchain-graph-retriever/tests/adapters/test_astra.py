from collections.abc import Iterator

import pytest
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_astradb.utils.vector_store_codecs import _DefaultVSDocumentCodec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.adapters.astra import AstraAdapter, _QueryHelper

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


class TestAstraAdapter(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(
        self,
        enabled_stores: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator["AstraAdapter"]:
        if "astra" not in enabled_stores:
            pytest.skip("Pass --stores=astra to test Astra")
            return

        import os

        from astrapy import AstraDBDatabaseAdmin
        from astrapy.authentication import StaticTokenProvider
        from dotenv import load_dotenv
        from langchain_astradb import AstraDBVectorStore
        from langchain_graph_retriever.adapters.astra import (
            AstraAdapter,
        )

        load_dotenv()

        token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
        keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
        api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

        admin = AstraDBDatabaseAdmin(api_endpoint=api_endpoint, token=token)
        admin.create_keyspace(keyspace)

        store = AstraDBVectorStore(
            embedding=animal_embeddings,
            collection_name="animals",
            namespace=keyspace,
            token=token,
            api_endpoint=api_endpoint,
            pre_delete_collection=True,
        )
        store.add_documents(animal_docs)

        yield AstraAdapter(store)

        store.delete_collection()
