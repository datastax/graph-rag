from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class KeybertKeywordExtractor(BaseDocumentTransformer):
    """
    Add metadata to documents about keywords using `KeyBERT <https://maartengr.github.io/KeyBERT/>`_.

    KeyBERT is a minimal and easy-to-use keyword extraction technique that
    leverages BERT embeddings to create keywords and keyphrases that are most
    similar to a document.

    The KeybertKeywordExtractor uses KeyBERT add a list of keywords to a
    document's metadata.

    Preliminaries
    -------------

    Install the ``keybert`` package.

    Note that ``bs4`` is also installed to support the WebBaseLoader in the example,
    but not needed by the KeybertKeywordExtractor itself.

    .. code-block:: bash

        pip install -q langchain_community bs4 keybert

    Example
    -------
    We load the ``state_of_the_union.txt`` file, chunk it, then for each chunk we
    add keywords to the metadata.

    .. code-block:: python

        from langchain_community.document_loaders import WebBaseLoader
        from langchain_community.document_transformers import KeybertKeywordExtractor
        from langchain_text_splitters import CharacterTextSplitter

        loader = WebBaseLoader(
            "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"
        )
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)

        extractor = KeybertKeywordExtractor()
        documents = extractor.transform_documents(documents)

        print(documents[0].metadata)

    .. code-block:: output

        {'source': 'https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt', 'keywords': ['putin', 'vladimir', 'ukrainian', 'russia', 'ukraine']}

    Parameters
    ----------
    batch_size : int, default 8
        The number of documents to process in each batch.
    metadata_key : str, default "keywords"
        The name of the key used in the metadata output.
    model : str, default "all-MiniLM-L6-v2"
        The KeyBERT model to use.
    """  # noqa: E501

    def __init__(
        self,
        *,
        batch_size: int = 8,
        metadata_key: str = "keywords",
        model: str = "all-MiniLM-L6-v2",
    ):
        try:
            import keybert  # type: ignore

            self._kw_model = keybert.KeyBERT(model=model)
        except ImportError:
            raise ImportError(
                "keybert is required for the KeybertLinkExtractor. "
                "Please install it with `pip install keybert`."
            ) from None

        self._batch_size = batch_size
        self._metadata_key = metadata_key

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            texts = [item.page_content for item in batch]
            extracted = self._kw_model.extract_keywords(docs=texts, **kwargs)
            if len(texts) == 1:
                # Even though we pass a list, if it contains one item, keybert will
                # flatten it. This means it's easier to just call the special case
                # for one item.
                batch[0].metadata[self._metadata_key] = [kw[0] for kw in extracted]
            else:
                for i, keywords in enumerate(extracted):
                    batch[i].metadata[self._metadata_key] = [kw[0] for kw in keywords]
        return documents
