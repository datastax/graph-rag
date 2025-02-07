import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.transformers.parent import ParentTransformer


def test_transform_documents():
    root = Document(id="root", page_content="test", metadata={"_path": "root"})
    h1 = Document(id="h1", page_content="test", metadata={"_path": "root.h1"})
    h1a = Document(id="h1a", page_content="test", metadata={"_path": "root.h1.a"})

    original_h1 = h1.model_copy()

    transformer = ParentTransformer(
        path_metadata_key="_path", parent_metadata_key="_parent", path_delimiter="."
    )

    transformed_root = transformer.transform_documents([root])[0]
    assert "_parent" not in transformed_root.metadata

    transformed_h1 = transformer.transform_documents([h1])[0]
    assert transformed_h1.metadata["_parent"] == "root"

    transformed_h1a = transformer.transform_documents([h1a])[0]
    assert transformed_h1a.metadata["_parent"] == "root.h1"

    transformer = ParentTransformer()
    with pytest.raises(ValueError, match="path not found in metadata"):
        transformer.transform_documents([root])

    # confirm original docs aren't modified
    assert original_h1 == h1
