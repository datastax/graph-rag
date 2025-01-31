from collections.abc import Sequence
from typing import Any

import spacy  # type: ignore
from langchain_core.documents import BaseDocumentTransformer, Document
from spacy.language import Language  # type: ignore
from typing_extensions import override


class SpacyNERTransformer(BaseDocumentTransformer):
    """
    Add metadata to documents about named entities using `spaCy`.

    Preliminaries
    -------------

    Install the ``spacy`` package. The below uses ``spacy[apple]`` so it includes optimizations
    for running on Apple M1 hardrware.

    .. code-block:: bash

        pip install -q "spacy[apple]"

    Download the model.

    .. code-block:: bash

        python -m spacy download en_core_web_sm

    Example
    -------
    We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
    add named entities to the metadata.

    .. code-block:: python

        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import CharacterTextSplitter
        from langchain_graph_retriever.document_transformers import SpacyNERTransformer

        loader = WebBaseLoader(
            "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
        )
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        extractor = SpacyNERTransformer()
        documents = extractor.transform_documents(documents)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt', 'person': ['president zelenskyy', 'vladimir putin']}

    Parameters
    ----------
    include_labels : set[str], optional
        Set of entity labels to include. Will include all labels if empty.
    exclude_labels : set[str], optional
        Set of entity lables to exclude. Will not exclude anything if empty.
    metadata_key : str, default ""
        The metadata key to store the extracted entities in.
    model : str, default "en_core_web_sm"
        The spacy model to use.

    Notes
    -----
     See spaCy docs for the selected model to determine what NER labels will be
     used. The default model
    (en_core_web_sm)[https://spacy.io/models/en#en_core_web_sm-labels] produces:
    CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL,
    ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART.

    """  # noqa: E501

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
