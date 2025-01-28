from collections.abc import Sequence
from typing import Any

from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class HierarchyLinkExtractor(BaseDocumentTransformer):
    r"""
    Extract links from a document hierarchy.

    Example
    -------

    .. code-block:: python

        # Given three paths (in this case, within the "Root" document):
        h1 = ["Root", "H1"]
        h1a = ["Root", "H1", "a"]
        h1b = ["Root", "H1", "b"]

        # Parent links `h1a` and `h1b` to `h1`.
        # Child links `h1` to `h1a` and `h1b`.
        # Sibling links `h1a` and `h1b` together (both directions).

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
    path_metadata_key : str, default "path"
        Metadata key containing the path.
        This may correspond to paths in a file system, hierarchy in a document, etc.
    path_delimiter : str, default "\"
        Delimiter of items in the path.
    parent_links : bool, default True
        Whether to link each document to it's parent.
        If `True`, `metadata["parent_out"]` will be populated to link to
        `metadata["parent_in"]`.
    child_links: bool, default False
        Whether to link each document from a section to its children.
        If `True`, `metadata["child_out"]` will be populated to link to
        `metadata["child_in"]`.
    sibling_links : bool, default False
        Whether to link each document to sibling (adjacent) documents.
        If `True`, `metadata["sibling"]` will be populated.
    """

    def __init__(
        self,
        *,
        path_metadata_key: str = "path",
        path_delimiter: str = "\\",
        parent_links: bool = True,
        child_links: bool = False,
        sibling_links: bool = False,
    ):
        self._path_metadata_key = path_metadata_key
        self._path_delimiter = path_delimiter
        self._parent_links = parent_links
        self._child_links = child_links
        self._sibling_links = sibling_links

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
            if self._parent_links:
                # This is linked from everything with this parent path.
                document.metadata["parent_in"] = path
            if self._child_links:
                # This is linked to every child with this as it's "parent" path.
                document.metadata["child_out"] = path

            path_parts = path.split(self._path_delimiter)
            if len(path_parts) >= 1:
                parent_path = self._path_delimiter.join(path_parts[0:-1])
                if self._parent_links and len(path_parts) > 1:
                    # This is linked to the nodes with the given parent path.
                    document.metadata["parent_out"] = parent_path
                if self._child_links and len(path_parts) > 1:
                    # This is linked from every node with the given parent path.
                    document.metadata["child_in"] = parent_path
                if self._sibling_links:
                    # This is a sibling of everything with the same parent.
                    document.metadata["sibling"] = parent_path
        return documents
