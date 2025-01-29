import json

from langchain_core.documents import Document
from langchain_graph_retriever.document_transformers.metadata_denormalizer import (
    DEFAULT_PATH_DELIMITER,
    DEFAULT_STATIC_VALUE,
    DENORMALIZED_KEYS_KEY,
    MetadataDenormalizer,
)


def test_transform_documents(animal_docs: list[Document]):
    original_docs = [animal_docs[0]]
    transformer = MetadataDenormalizer()

    # confirm "keywords" contains a list value in original document
    list_key = "keywords"
    original_metadata = original_docs[0].metadata
    assert isinstance(original_metadata[list_key], list)

    # pull the first keyword in that list
    first_keyword = original_metadata[list_key][0]

    # transform the document
    denormalized_docs = transformer.transform_documents(original_docs)
    denormalized_metadata = denormalized_docs[0].metadata

    # confirm "keywords" no longer exists as a metadata key
    assert list_key not in denormalized_metadata

    # confirm "keywords" exists as a denormalized key
    assert list_key in json.loads(denormalized_metadata[DENORMALIZED_KEYS_KEY])

    # confirm the denormalized key has the expected value
    denormalized_key = f"{list_key}{DEFAULT_PATH_DELIMITER}{first_keyword}"
    assert denormalized_key in denormalized_metadata
    assert denormalized_metadata[denormalized_key] == DEFAULT_STATIC_VALUE


def test_revert_documents(animal_docs: list[Document]):
    original_docs = [animal_docs[0]]
    transformer = MetadataDenormalizer()

    denormalized_docs = transformer.transform_documents(original_docs)

    reverted_docs = transformer.revert_documents(denormalized_docs)

    assert original_docs[0] == reverted_docs[0]
