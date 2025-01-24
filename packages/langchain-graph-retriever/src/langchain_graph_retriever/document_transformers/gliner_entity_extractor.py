from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document


class GLiNEREntityExtractor(BaseDocumentTransformer):
    """Add metadata to documents about named entities using `GLiNER`_.

    `GLiNER`_ is a Named Entity Recognition (NER) model capable of identifying any
    entity type using a bidirectional transformer encoder (BERT-like).

    Preliminaries
    -------------

    Install the ``gliner`` package.

    Note that ``bs4`` is also installed to support the WebBaseLoader in the example,
    but not needed by the GLiNEREntityExtractor itself.

    .. code-block:: bash

        pip install -q langchain_community bs4 gliner

    Example:
    -------
    We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
    add named entities to the metadata.

    .. code-block:: python

        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.document_transformers import GLiNEREntityExtractor
        from langchain_text_splitters import CharacterTextSplitter

        loader = WebBaseLoader(
            "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
        )
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        extractor = GLiNEREntityExtractor(labels=["person", "topic"])
        documents = extractor.transform_documents(documents)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt', 'person': ['president zelenskyy', 'vladimir putin']}

    Args:
        labels: List of entity kinds to extract.
        batch_size: The number of documents to process in each batch (default ``8``)
        metadata_key_prefix: A prefix to add to metadata keys outputted by the extractor (default ``""``)
        model: The GLiNER model to use. (default ``urchade/gliner_mediumv2.1``)

    """  # noqa: E501

    def __init__(
        self,
        labels: list[str],
        *,
        batch_size: int = 8,
        metadata_key_prefix: str = "",
        model: str = "urchade/gliner_mediumv2.1",
    ):
        try:
            from gliner import GLiNER  # type: ignore

            self._model = GLiNER.from_pretrained(model)

        except ImportError:
            raise ImportError(
                "gliner is required for the GLiNEREntityExtractor. "
                "Please install it with `pip install gliner`."
            ) from None

        self._batch_size = batch_size
        self._labels = labels
        self.metadata_key_prefix = metadata_key_prefix

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extract named entities from documents using GLiNER.

        Args:
            documents: The sequence of documents to transform
            kwargs: Keyword arguments to pass to GLiNER. See: https://github.com/urchade/GLiNER/blob/v0.2.13/gliner/model.py#L419-L421

        """
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._model.batch_predict_entities(
                texts=texts, labels=self._labels, **kwargs
            )
            for i, entities in enumerate(extracted):
                labels = set()
                for entity in entities:
                    label = self.metadata_key_prefix + entity["label"]
                    labels.add(label)
                    batch[i].metadata.setdefault(label, set()).add(
                        entity["text"].lower()
                    )
                for label in labels:
                    batch[i].metadata[label] = list(batch[i].metadata[label])
        return documents
