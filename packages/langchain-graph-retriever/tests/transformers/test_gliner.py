from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.transformers.gliner import GLiNERTransformer


@pytest.mark.extra
def test_transform_documents(animal_docs: list[Document]):
    from gliner import GLiNER  # type: ignore

    class FakeGLiNER(GLiNER):
        def __init__(self):
            pass

        def batch_predict_entities(
            self, texts: list[str], **kwargs: Any
        ) -> list[list[dict[str, str]]]:
            return [[{"text": text.split()[0], "label": "first"}] for text in texts]

    fake_model = FakeGLiNER()
    transformer = GLiNERTransformer(
        ["first"], model=fake_model, metadata_key_prefix="prefix_"
    )

    transformed_docs = transformer.transform_documents(animal_docs)
    assert "prefix_first" in transformed_docs[0].metadata

    with pytest.raises(ValueError, match="Invalid model"):
        GLiNERTransformer([], model={})
