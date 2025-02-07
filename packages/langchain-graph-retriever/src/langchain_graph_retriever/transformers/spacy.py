from collections.abc import Sequence
from typing import Any

import spacy  # type: ignore
from langchain_core.documents import BaseDocumentTransformer, Document
from spacy.language import Language  # type: ignore
from typing_extensions import override


class SpacyNERTransformer(BaseDocumentTransformer):
    """
    Add metadata to documents about named entities using **spaCy**.

    Identifies and labels named entities in text, extracting structured
    metadata such as organizations, locations, dates, and other key entity types.

    [**spaCy**](https://spacy.io/) is a library for Natural Language Processing
    in Python. Here it is used for Named Entity Recognition (NER) to extract values
    from document content.

    Prerequisites
    -------------

    This transformer requires the `spacy` extra to be installed.

    ```
    pip install -qU langchain_graph_retriever[spacy]
    ```

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#spacynertransformer) in the guide.

    Parameters
    ----------
    include_labels :
        Set of entity labels to include. Will include all labels if empty.
    exclude_labels :
        Set of entity labels to exclude. Will not exclude anything if empty.
    metadata_key :
        The metadata key to store the extracted entities in.
    model :
        The spaCy model to use. Pass the name of a model to load
        or pass an instantiated spaCy model instance.

    Notes
    -----
     See spaCy docs for the selected model to determine what NER labels will be
     used. The default model
    [en_core_web_sm](https://spacy.io/models/en#en_core_web_sm-labels) produces:
    CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL,
    ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART.

    """

    def __init__(
        self,
        *,
        include_labels: set[str] = set(),
        exclude_labels: set[str] = set(),
        limit: int | None = None,
        metadata_key: str = "entities",
        model: str | Language = "en_core_web_sm",
    ):
        self.include_labels = include_labels
        self.exclude_labels = exclude_labels
        self.limit = limit
        self.metadata_key = metadata_key

        if isinstance(model, str):
            if not spacy.util.is_package(model):
                spacy.cli.download(model)  # type: ignore
            self.model = spacy.load(model)
        elif isinstance(model, Language):
            self.model = model
        else:
            raise ValueError(f"Invalid model: {model}")

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        docs = []
        for doc in documents:
            results = self.model(doc.page_content).ents
            # Filter and de-duplicate entities.
            entities = list(
                {
                    f"{e.label_}: {e.text}"
                    for e in results
                    if not self.include_labels or e.label_ in self.include_labels
                    if not self.exclude_labels or e.label_ not in self.exclude_labels
                }
            )
            # Limit it, if necessary.
            if self.limit:
                entities = entities[: self.limit]
            docs.append(
                Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata={self.metadata_key: entities, **doc.metadata},
                )
            )
        return docs
