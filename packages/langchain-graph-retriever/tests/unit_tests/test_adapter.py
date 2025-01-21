from langchain_graph_retriever.adapters import Adapter
from langchain_graph_retriever.edge_helper import Edge


class FakeAdapter(Adapter):
    def get(self, **kwargs):
        pass

    def similarity_search_with_embedding_by_vector(self, **kwargs):
        pass


def test_get_metadata_filter_normalized():
    adapter = FakeAdapter([], use_normalized_metadata=True)

    assert adapter._get_metadata_filter(edge=Edge(key="boolean", value=True)) == {
        "boolean": True
    }

    assert adapter._get_metadata_filter(edge=Edge(key="incoming", value=4)) == {
        "incoming": 4
    }

    assert adapter._get_metadata_filter(edge=Edge(key="place", value="berlin")) == {
        "place": "berlin"
    }


def test_get_metadata_filter_denormalized() -> None:
    adapter = FakeAdapter([], use_normalized_metadata=False)

    assert adapter._get_metadata_filter(edge=Edge(key="boolean", value=True)) == {
        "boolean": True
    }

    assert adapter._get_metadata_filter(
        edge=Edge(key="incoming", value=4), denormalize_edge=True
    ) == {"incoming.4": "$"}

    assert adapter._get_metadata_filter(
        edge=Edge(key="place", value="berlin"), denormalize_edge=True
    ) == {"place.berlin": "$"}
