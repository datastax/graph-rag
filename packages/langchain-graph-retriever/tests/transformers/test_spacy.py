import pytest
from langchain_core.documents import Document
from langchain_graph_retriever.transformers.spacy import SpacyNERTransformer


@pytest.mark.extra
def test_transform_documents(animal_docs: list[Document]):
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    from spacy.vocab import Vocab

    class FakeLanguage(Language):
        def __init__(self):
            pass

        def __call__(self, text: str | Doc, **kwargs) -> Doc:
            vocab = Vocab()
            assert isinstance(text, str)
            doc = Doc(vocab=vocab, words=text.split())
            doc.ents = [
                Span(doc, start=0, end=1, label="first"),
                Span(doc, start=1, end=2, label="second"),
                Span(doc, start=2, end=3, label="third"),
            ]
            return doc

    fake_model = FakeLanguage()

    transformer = SpacyNERTransformer(model=fake_model, metadata_key="spacey")
    transformed_docs = transformer.transform_documents(animal_docs)
    assert "spacey" in transformed_docs[0].metadata
    assert "first: the" in transformed_docs[0].metadata["spacey"]
    assert "second: aardvark" in transformed_docs[0].metadata["spacey"]
    assert "third: is" in transformed_docs[0].metadata["spacey"]

    transformer = SpacyNERTransformer(
        model=fake_model, metadata_key="spacey", include_labels=set(["first"])
    )
    transformed_docs = transformer.transform_documents(animal_docs)
    assert "spacey" in transformed_docs[0].metadata
    assert "first: the" in transformed_docs[0].metadata["spacey"]
    assert "second: aardvark" not in transformed_docs[0].metadata["spacey"]
    assert "third: is" not in transformed_docs[0].metadata["spacey"]

    transformer = SpacyNERTransformer(
        model=fake_model, metadata_key="spacey", exclude_labels=set(["first"])
    )
    transformed_docs = transformer.transform_documents(animal_docs)
    assert "spacey" in transformed_docs[0].metadata
    assert "first: the" not in transformed_docs[0].metadata["spacey"]
    assert "second: aardvark" in transformed_docs[0].metadata["spacey"]
    assert "third: is" in transformed_docs[0].metadata["spacey"]

    transformer = SpacyNERTransformer(model=fake_model, metadata_key="spacey", limit=1)
    transformed_docs = transformer.transform_documents(animal_docs)
    assert len(transformed_docs[0].metadata["spacey"]) == 1

    with pytest.raises(ValueError, match="Invalid model"):
        SpacyNERTransformer(model={})  # type: ignore
