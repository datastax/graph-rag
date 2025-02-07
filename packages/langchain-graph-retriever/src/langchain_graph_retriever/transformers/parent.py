from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class ParentTransformer(BaseDocumentTransformer):
    """
    Adds the hierarchal Parent path to the document metadata.

    Parameters
    ----------
    path_metadata_key :
        Metadata key containing the path.
        This may correspond to paths in a file system, hierarchy in a document, etc.
    parent_metadata_key:
        Metadata key for the added parent path
    path_delimiter :
        Delimiter of items in the path.

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#parenttransformer) in the guide.

    Notes
    -----
    Expects each document to contain its _path_ in its metadata.
    """

    def __init__(
        self,
        *,
        path_metadata_key: str = "path",
        parent_metadata_key: str = "parent",
        path_delimiter: str = "\\",
    ):
        self._path_metadata_key = path_metadata_key
        self._parent_metadata_key = parent_metadata_key
        self._path_delimiter = path_delimiter

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        results: list[Document] = []
        for document in documents:
            if self._path_metadata_key not in document.metadata:
                msg = (
                    f"path not found in metadata at {self._path_metadata_key}",
                    f" for document id: {document.id}",
                )
                raise ValueError(msg)

            path: str = document.metadata[self._path_metadata_key]
            path_parts = path.split(self._path_delimiter)
            result = document.model_copy()
            if len(path_parts) > 1:
                parent_path = self._path_delimiter.join(path_parts[0:-1])
                result.metadata[self._parent_metadata_key] = parent_path
            results.append(result)
        return results
