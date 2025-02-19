from collections.abc import Iterable

import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TestOpenSearch(AdapterComplianceSuite):
    def supports_nested_metadata(self) -> bool:
        return False

    def supports_dict_in_list(self) -> bool:
        return False

    @pytest.fixture(scope="class")
    def adapter(
        self,
        request: pytest.FixtureRequest,
        enabled_stores: set[str],
        testcontainers: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterable[Adapter]:
        if "opensearch" not in enabled_stores:
            pytest.skip("Pass --stores=opensearch to test OpenSearch")

        from langchain_community.vectorstores import OpenSearchVectorSearch
        from langchain_graph_retriever.adapters.open_search import (
            OpenSearchAdapter,
        )

        if "opensearch" in testcontainers:
            from testcontainers.opensearch import OpenSearchContainer  # type: ignore

            # If the admin password doesn't pass the length and regex requirements
            # starting the container will hang (`docker ps <container_id>` to debug).
            container = OpenSearchContainer(
                image="opensearchproject/opensearch:2.18.0",
                initial_admin_password="SomeRandomP4ssword",
            )
            container.start()
            request.addfinalizer(lambda: container.stop())

            config = container.get_config()
            opensearch_url = f"http://{config['host']}:{config['port']}"
            kwargs = {"http_auth": (config["username"], config["password"])}
        else:
            opensearch_url = "http://localhost:9200"
            kwargs = {}

        store = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name="animals",
            embedding_function=animal_embeddings,
            engine="faiss",
            **kwargs,
        )
        store.add_documents(animal_docs)

        yield OpenSearchAdapter(store)

        if store.index_exists():
            store.delete_index()
