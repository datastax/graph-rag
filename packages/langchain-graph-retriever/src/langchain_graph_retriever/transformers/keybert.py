from collections.abc import Sequence
from typing import Any

from keybert import KeyBERT  # type: ignore
from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class KeyBERTTransformer(BaseDocumentTransformer):
    """
    Add metadata to documents about keywords using **KeyBERT**.

    Extracts key topics and concepts from text, generating metadata that highlights
    the most relevant terms to describe the content.

    [**KeyBERT**](https://maartengr.github.io/KeyBERT) is a minimal and easy-to-use
    keyword extraction technique that leverages BERT embeddings to create keywords and
    keyphrases that are most similar to a document.

    Prerequisites
    -------------

    This transformer requires the `keybert` extra to be installed.

    ```
    pip install -qU langchain_graph_retriever[keybert]
    ```

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#keyberttransformer) in the guide.

    Parameters
    ----------
    batch_size
        The number of documents to process in each batch.
    metadata_key :
        The name of the key used in the metadata output.
    model :
        The KeyBERT model to use. Pass the name of a model to load
        or pass an instantiated KeyBERT model instance.
    """

    def __init__(
        self,
        *,
        batch_size: int = 8,
        metadata_key: str = "keywords",
        model: str | KeyBERT = "all-MiniLM-L6-v2",
    ):
        if isinstance(model, KeyBERT):
            self._kw_model = model
        elif isinstance(model, str):
            self._kw_model = KeyBERT(model=model)
        else:
            raise ValueError(f"Invalid model: {model}")
        self._batch_size = batch_size
        self._metadata_key = metadata_key

    def _extract_keywords(
        self, docs: list[str], **kwargs
    ) -> list[list[tuple[str, float]]]:
        """Wrap the function to always return a list of responses."""
        extracted = self._kw_model.extract_keywords(docs=docs, **kwargs)
        if len(docs) == 1:
            # Even if we pass a list, if it contains one item, keybert will flatten it.
            return [extracted]
        else:
            return extracted

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        results: list[Document] = []
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._extract_keywords(docs=texts, **kwargs)
            for j, keywords in enumerate(extracted):
                results.append(
                    Document(
                        id=batch[j].id,
                        page_content=batch[j].page_content,
                        metadata={
                            self._metadata_key: [kw[0] for kw in keywords],
                            **batch[j].metadata,
                        },
                    )
                )
        return results
