from typing import Iterable

from langchain_core.documents import Document


class DocumentCache:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    def add_document(self, doc: Document, depth_found: int) -> None:
        if doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        if doc.id not in self.documents:
            doc.metadata["_depth_found"] = depth_found
            self.documents[doc.id] = doc

    def add_documents(self, docs: Iterable[Document], depth_found: int) -> None:
        for doc in docs:
            self.add_document(doc=doc, depth_found=depth_found)

    def get_by_document_ids(
        self,
        ids: Iterable[str],
    ) -> list[Document]:
        docs: list[Document] = []
        for id in ids:
            if id in self.documents:
                docs.append(self.documents[id])
            else:
                msg = f"unexpected, cache should contain id: {id}"
                raise RuntimeError(msg)
        return docs
