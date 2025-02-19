import json

from langchain_core.documents import Document
from langchain_graph_retriever.transformers.shredding import (
    DEFAULT_PATH_DELIMITER,
    DEFAULT_STATIC_VALUE,
    SHREDDED_KEYS_KEY,
    ShreddingTransformer,
)


def test_transform_documents(animal_docs: list[Document]):
    first_doc = animal_docs[0].model_copy()
    original_docs = [animal_docs[0]]
    transformer = ShreddingTransformer()

    # confirm "keywords" contains a list value in original document
    list_key = "keywords"
    original_metadata = original_docs[0].metadata
    assert isinstance(original_metadata[list_key], list)

    # pull the first keyword in that list
    first_keyword = original_metadata[list_key][0]

    # transform the document
    shredded_docs = transformer.transform_documents(original_docs)
    shredded_metadata = shredded_docs[0].metadata

    # confirm "keywords" no longer exists as a metadata key
    assert list_key not in shredded_metadata

    # confirm "keywords" exists as a shredded key
    assert list_key in json.loads(shredded_metadata[SHREDDED_KEYS_KEY])

    # confirm the shredded key has the expected value
    shredded_key = f'{list_key}{DEFAULT_PATH_DELIMITER}"{first_keyword}"'
    assert shredded_key in shredded_metadata
    assert shredded_metadata[shredded_key] == DEFAULT_STATIC_VALUE

    # confirm original docs aren't modified
    assert first_doc == animal_docs[0]


def test_restore_documents(animal_docs: list[Document]):
    first_doc = animal_docs[0].model_copy()
    original_docs = [animal_docs[0]]
    transformer = ShreddingTransformer()

    shredded_docs = transformer.transform_documents(original_docs)

    restored_docs = transformer.restore_documents(shredded_docs)

    assert original_docs[0] == restored_docs[0]

    # confirm original docs aren't modified
    assert first_doc == animal_docs[0]
