"""Shredding Transformer for sequence-based metadata fields."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override

SHREDDED_KEYS_KEY = "__shredded_keys"
DEFAULT_PATH_DELIMITER = "→"  # unicode 2192
DEFAULT_STATIC_VALUE = "§"  # unicode 00A7


class ShreddingTransformer(BaseDocumentTransformer):
    """
    Shreds sequence-based metadata fields.

    Certain vector stores do not support storing or searching on metadata fields
    with sequence-based values. This transformer converts sequence-based fields
    into simple metadata values.

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#shreddingtransformer) in the guide.

    Parameters
    ----------
    keys :
        A set of metadata keys to shred.
        If empty, all sequence-based fields will be shredded.
    path_delimiter :
        The path delimiter to use when building shredded keys.
    static_value :
        The value to set on each shredded key.
    """

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
            shredded_keys: list[str] = []
            for key, value in document.metadata.items():
                is_nested_sequence = isinstance(value, Sequence) and not isinstance(
                    value, str | bytes
                )
                should_shred = (not self.keys) or (key in self.keys)
                if is_nested_sequence and should_shred:
                    shredded_keys.append(key)
                    for item in value:
                        new_doc.metadata[self.shredded_key(key=key, value=item)] = (
                            self.shredded_value()
                        )
                else:
                    new_doc.metadata[key] = value
            if len(shredded_keys) > 0:
                new_doc.metadata[SHREDDED_KEYS_KEY] = json.dumps(shredded_keys)
            transformed_docs.append(new_doc)

        return transformed_docs

    def restore_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """
        Restore documents transformed by the ShreddingTransformer.

        Restore documents transformed by the ShreddingTransformer back to
        their original state before shredding.

        Note that any non-string values inside lists will be converted to strings
        after restoring.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns
        -------
        Sequence[Document]
            A sequence of transformed Documents.
        """
        restored_docs = []
        for document in documents:
            new_doc = Document(id=document.id, page_content=document.page_content)
            shredded_keys = set(
                json.loads(document.metadata.pop(SHREDDED_KEYS_KEY, "[]"))
            )

            for key, value in document.metadata.items():
                # Check if the key belongs to a shredded group
                split_key = key.split(self.path_delimiter, 1)
                if (
                    len(split_key) == 2
                    and split_key[0] in shredded_keys
                    and value == self.static_value
                ):
                    original_key, original_value = split_key
                    value = json.loads(original_value)
                    if original_key not in new_doc.metadata:
                        new_doc.metadata[original_key] = []
                    new_doc.metadata[original_key].append(value)
                else:
                    # Retain non-shredded metadata as is
                    new_doc.metadata[key] = value

            restored_docs.append(new_doc)

        return restored_docs

    def shredded_key(self, key: str, value: Any) -> str:
        """
        Get the shredded key for a key/value pair.

        Parameters
        ----------
        key :
            The metadata key to shred
        value :
            The metadata value to shred

        Returns
        -------
        str
            the shredded key
        """
        return f"{key}{self.path_delimiter}{json.dumps(value)}"

    def shredded_value(self) -> str:
        """
        Get the shredded value for a key/value pair.

        Returns
        -------
        str
            the shredded value
        """
        return self.static_value
