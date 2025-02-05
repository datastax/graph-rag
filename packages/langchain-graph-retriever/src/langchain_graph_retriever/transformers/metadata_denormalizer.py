"""Denormalizer for sequence-based metadata fields."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override

DENORMALIZED_KEYS_KEY = "__denormalized_keys"
DEFAULT_PATH_DELIMITER = "→"  # unicode 2192
DEFAULT_STATIC_VALUE = "§"  # unicode 00A7


class MetadataDenormalizer(BaseDocumentTransformer):
    """
    Denormalizes sequence-based metadata fields.

    Certain vector stores do not support storing or searching on metadata fields
    with sequence-based values. This transformer converts sequence-based fields
    into simple metadata values.

    Example:
    -------

    .. code-block:: python

        from langchain_core.documents import Document
        from langchain_community.document_transformers.metadata_denormalizer import (
            MetadataDenormalizer,
        )

        doc = Document(
            page_content="test",
            metadata={"place": ["berlin", "paris"], "topic": ["weather"]},
        )

        de_normalizer = MetadataDenormalizer()

        docs = de_normalizer.transform_documents([doc])

        print(docs[0].metadata)


    .. code-block:: output

        {'place.berlin': True, 'place.paris': True, 'topic.weather': True}

    Parameters
    ----------
    keys : set[str], optional:
        A set of metadata keys to denormalize.
        If empty, all sequence-based fields will be denormalized.
    path_delimiter : str, default "→" (unicode 2192)
        The path delimiter to use when building denormalized keys.
    static_value : str, default "§" (unicode 00A7)
        The value to set on each denormalized key.

    """  # noqa: E501

    def __init__(
        self,
        *,
        keys: set[str] = set(),
        path_delimiter: str = DEFAULT_PATH_DELIMITER,
        static_value: Any = DEFAULT_STATIC_VALUE,
    ):
        self.keys = keys
        self.path_delimiter = path_delimiter
        self.static_value = static_value

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        transformed_docs = []
        for document in documents:
            new_doc = Document(id=document.id, page_content=document.page_content)
            denormalized_keys: list[str] = []
            for key, value in document.metadata.items():
                is_normalized = isinstance(value, Sequence) and not isinstance(
                    value, str | bytes
                )
                should_denormalize = (not self.keys) or (key in self.keys)
                if is_normalized and should_denormalize:
                    denormalized_keys.append(key)
                    for item in value:
                        new_doc.metadata[self.denormalized_key(key=key, value=item)] = (
                            self.denormalized_value()
                        )
                else:
                    new_doc.metadata[key] = value
            if len(denormalized_keys) > 0:
                new_doc.metadata[DENORMALIZED_KEYS_KEY] = json.dumps(denormalized_keys)
            transformed_docs.append(new_doc)

        return transformed_docs

    def revert_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """
        Revert documents transformed by the MetadataDenormalizer.

        Reverts documents transformed by the MetadataDenormalizer back to their original
        state before denormalization.

        Note that any non-string values inside lists will be converted to strings
        after reverting.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns
        -------
        Sequence[Document]
            A sequence of transformed Documents.
        """
        reverted_docs = []
        for document in documents:
            new_doc = Document(id=document.id, page_content=document.page_content)
            denormalized_keys = set(
                json.loads(document.metadata.pop(DENORMALIZED_KEYS_KEY, "[]"))
            )

            for key, value in document.metadata.items():
                # Check if the key belongs to a denormalized group
                split_key = key.split(self.path_delimiter, 1)
                if (
                    len(split_key) == 2
                    and split_key[0] in denormalized_keys
                    and value == self.static_value
                ):
                    original_key, original_value = split_key
                    if original_key not in new_doc.metadata:
                        new_doc.metadata[original_key] = []
                    new_doc.metadata[original_key].append(original_value)
                else:
                    # Retain non-denormalized metadata as is
                    new_doc.metadata[key] = value

            reverted_docs.append(new_doc)

        return reverted_docs

    def denormalized_key(self, key: str, value: Any) -> str:
        """
        Get the denormalized key for a key/value pair.

        Parameters
        ----------
        key : str
            The metadata key to denormalize
        value : Any
            The metadata value to denormalize

        Returns
        -------
        str
            the denormalized key
        """
        return f"{key}{self.path_delimiter}{value}"

    def denormalized_value(self) -> str:
        """
        Get the denormalized value for a key/value pair.

        Returns
        -------
        str
            the denormalized value
        """
        return self.static_value
