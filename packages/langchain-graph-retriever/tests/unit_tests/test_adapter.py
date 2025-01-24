from langchain_graph_retriever.adapters import Adapter
from langchain_graph_retriever.types import MetadataEdge


class FakeAdapter(Adapter):
    def get(self, **kwargs):
        pass

    def similarity_search_with_embedding_by_vector(self, **kwargs):
        pass


def test_get_metadata_filter_normalized():
    adapter = FakeAdapter([], use_normalized_metadata=True)

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="boolean", value=True)
    ) == {"boolean": True}

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="incoming", value=4)
    ) == {"incoming": 4}

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="place", value="berlin")
    ) == {"place": "berlin"}


def test_get_metadata_filter_denormalized() -> None:
    adapter = FakeAdapter([], use_normalized_metadata=False)

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="boolean", value=True)
    ) == {"boolean": True}

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="incoming", value=4), denormalize_edge=True
    ) == {"incoming.4": "$"}

    assert adapter._get_metadata_filter(
        edge=MetadataEdge(incoming_field="place", value="berlin"), denormalize_edge=True
    ) == {"place.berlin": "$"}
