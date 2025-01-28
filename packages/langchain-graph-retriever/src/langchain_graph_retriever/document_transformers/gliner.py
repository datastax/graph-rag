from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class GLiNEREntityExtractor(BaseDocumentTransformer):
    """
    Add metadata to documents about named entities using `GLiNER`_.

    `GLiNER`_ is a Named Entity Recognition (NER) model capable of identifying any
    entity type using a bidirectional transformer encoder (BERT-like).

    Preliminaries
    -------------

    Install the ``gliner`` package.

    Note that ``bs4`` is also installed to support the WebBaseLoader in the example,
    but not needed by the GLiNEREntityExtractor itself.

    .. code-block:: bash

        pip install -q langchain_community bs4 gliner

    Example
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

    Parameters
    ----------
    labels : list[str]
        List of entity kinds to extract.
    batch_size : int, default 8
        The number of documents to process in each batch.
    metadata_key_prefix : str, default ""
        A prefix to add to metadata keys outputted by the extractor.
        This will be prepended to the label, with the value (or values) holding the
        generated keywords for that entity kind.
    model : str, default "urchade/gliner_mediumv2.1"
        The GLiNER model to use.

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

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
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
