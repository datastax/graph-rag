from typing import Any

import pytest
from langchain_core.documents import Document


@pytest.mark.extra
def test_transform_documents(animal_docs: list[Document]):
    from keybert import KeyBERT  # type: ignore
    from langchain_graph_retriever.transformers.keybert import KeyBERTTransformer

    class FakeKeyBERT(KeyBERT):
        def __init__(self):
            pass

        def extract_keywords(
            self, docs: list[str], **kwargs: Any
        ) -> list[list[tuple[str, float]]]:
            return [
                [(word, len(word)) for word in set(doc.split()) if len(word) > 5]
                for doc in docs
            ]

    first_doc = animal_docs[0].model_copy()

    fake_model = FakeKeyBERT()
    transformer = KeyBERTTransformer(model=fake_model, metadata_key="keybert")

    transformed_docs = transformer.transform_documents(animal_docs)
    assert "keybert" in transformed_docs[0].metadata

    with pytest.raises(ValueError, match="Invalid model"):
        KeyBERTTransformer(model={})

    # confirm original docs aren't modified
    assert first_doc == animal_docs[0]
