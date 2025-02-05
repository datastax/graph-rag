from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class ParentTransformer(BaseDocumentTransformer):
    r"""
    Adds the parent path to the document metadata.

    Example
    -------
    ```
    # Given document chunks with 4 different paths:
    root = Document(id="root", page_content="test", metadata={"path": "root"})
    h1 = Document(id="h1", page_content="test", metadata={"path": "root/h1"})
    h1a = Document(id="h1a", page_content="test", metadata={"path": "root/h1/a"})
    h1b = Document(id="h1b", page_content="test", metadata={"path": "root/h1/b"})
    ```

    Example use with documents
    --------------------------
    .. code_block: python
        transformer = LinkExtractorTransformer([
            HierarchyLinkExtractor().as_document_extractor(
                # Assumes the "path" to each document is in the metadata.
                # Could split strings, etc.
                lambda doc: doc.metadata.get("path", [])
            )
        ])
        linked = transformer.transform_documents(docs)

    Parameters
    ----------
    path_metadata_key :
        Metadata key containing the path.
        This may correspond to paths in a file system, hierarchy in a document, etc.
    parent_metadata_key: str, default "parent"
        Metadata key for the added parent path
    path_delimiter :
        Delimiter of items in the path.
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
        for document in documents:
            if self._path_metadata_key not in document.metadata:
                msg = (
                    f"path not found in metadata at {self._path_metadata_key}",
                    f" for document id: {document.id}",
                )
                raise ValueError(msg)

            path: str = document.metadata[self._path_metadata_key]
            path_parts = path.split(self._path_delimiter)
            if len(path_parts) > 1:
                parent_path = self._path_delimiter.join(path_parts[0:-1])
                document.metadata[self._parent_metadata_key] = parent_path
        return documents
