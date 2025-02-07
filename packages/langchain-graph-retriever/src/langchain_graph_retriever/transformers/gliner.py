from collections.abc import Sequence
from typing import Any

from gliner import GLiNER  # type: ignore
from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class GLiNERTransformer(BaseDocumentTransformer):
    """
    Add metadata to documents about named entities using **GLiNER**.

    Extracts structured entity labels from text, identifying key attributes and
    categories to enrich document metadata with semantic information.

    [**GLiNER**](https://github.com/urchade/GLiNER) is a Named Entity
    Recognition (NER) model capable of identifying any entity type using a
    bidirectional transformer encoder (BERT-like).

    Prerequisites
    -------------

    This transformer requires the `gliner` extra to be installed.

    ```
    pip install -qU langchain_graph_retriever[gliner]
    ```

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#glinertransformer) in the guide.


    Parameters
    ----------
    labels :
        List of entity kinds to extract.
    batch_size :
        The number of documents to process in each batch.
    metadata_key_prefix :
        A prefix to add to metadata keys outputted by the extractor.
        This will be prepended to the label, with the value (or values) holding the
        generated keywords for that entity kind.
    model :
        The GLiNER model to use. Pass the name of a model to load or
        pass an instantiated GLiNER model instance.

    """  # noqa: E501

    def __init__(
        self,
        labels: list[str],
        *,
        batch_size: int = 8,
        metadata_key_prefix: str = "",
        model: str | GLiNER = "urchade/gliner_mediumv2.1",
    ):
        if isinstance(model, GLiNER):
            self._model = model
        elif isinstance(model, str):
            self._model = GLiNER.from_pretrained(model)
        else:
            raise ValueError(f"Invalid model: {model}")

        self._batch_size = batch_size
        self._labels = labels
        self.metadata_key_prefix = metadata_key_prefix

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        results: list[Document] = []
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._model.batch_predict_entities(
                texts=texts, labels=self._labels, **kwargs
            )
            for j, entities in enumerate(extracted):
                new_metadata: dict[str, Any] = {}
                for entity in entities:
                    label = self.metadata_key_prefix + entity["label"]
                    new_metadata.setdefault(label, set()).add(entity["text"].lower())

                result = Document(
                    id=batch[j].id,
                    page_content=batch[j].page_content,
                    metadata=batch[j].metadata.copy(),
                )
                for key in new_metadata.keys():
                    result.metadata[key] = list(new_metadata[key])

                results.append(result)
        return results
