from typing import Any, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document


class HierarchyLinkExtractor(BaseDocumentTransformer):
    def __init__(
        self,
        *,
        path_metadata_key: str = "path",
        path_delimiter: str = "\\",
        parent_links: bool = True,
        child_links: bool = False,
        sibling_links: bool = False,
    ):
        """Extract links from a document hierarchy.

        Example:

            .. code-block:: python

                # Given three paths (in this case, within the "Root" document):
                h1 = ["Root", "H1"]
                h1a = ["Root", "H1", "a"]
                h1b = ["Root", "H1", "b"]

                # Parent links `h1a` and `h1b` to `h1`.
                # Child links `h1` to `h1a` and `h1b`.
                # Sibling links `h1a` and `h1b` together (both directions).

        Example use with documents:
            .. code_block: python
                transformer = LinkExtractorTransformer([
                    HierarchyLinkExtractor().as_document_extractor(
                        # Assumes the "path" to each document is in the metadata.
                        # Could split strings, etc.
                        lambda doc: doc.metadata.get("path", [])
                    )
                ])
                linked = transformer.transform_documents(docs)

        Args:
            kind: Kind of links to produce with this extractor.
            parent_links: Link from a section to its parent.
            child_links: Link from a section to its children.
            sibling_links: Link from a section to other sections with the same parent.
        """
        self._path_metadata_key = path_metadata_key
        self._path_delimiter = path_delimiter
        self._parent_links = parent_links
        self._child_links = child_links
        self._sibling_links = sibling_links

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extracts hyperlinks from html documents using BeautifulSoup4"""
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
