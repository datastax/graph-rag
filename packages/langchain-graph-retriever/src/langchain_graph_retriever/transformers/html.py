from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag
from langchain_core.documents import BaseDocumentTransformer, Document
from typing_extensions import override


class HyperlinkTransformer(BaseDocumentTransformer):
    """
    Extracts hyperlinks from HTML content and stores them in document metadata.

    Prerequisites
    -------------

    This transformer requires the `html` extra to be installed.

    ```
    pip install -qU langchain_graph_retriever[html]
    ```

    Example
    -------
    An example of how to use this transformer exists
    [HERE](../../guide/transformers.md#hyperlinktransformer) in the guide.

    Parameters
    ----------
    url_metadata_key :
        The metadata field containing the URL of the document. Must be set
        before transforming. Needed to resolve relative paths.
    metadata_key :
        The metadata field to populate with documents linked from this content.
    drop_fragments :
        Whether fragments in URLs and links should be dropped.

    Notes
    -----
    Expects each document to contain its _URL_ in its metadata.

    """  # noqa: E501

    def __init__(
        self,
        *,
        url_metadata_key: str = "url",
        metadata_key: str = "hyperlink",
        drop_fragments: bool = True,
    ):
        self._url_metadata_key = url_metadata_key
        self._metadata_key = metadata_key
        self._drop_fragments = drop_fragments

    @staticmethod
    def _parse_url(link: Tag, page_url: str, drop_fragments: bool = True) -> str | None:
        href = link.get("href")
        if href is None:
            return None
        if isinstance(href, list) and len(href) == 1:
            href = href[0]
        if not isinstance(href, str):
            return None

        url = urlparse(href)
        if url.scheme not in ["http", "https", ""]:
            return None

        # Join the HREF with the page_url to convert relative paths to absolute.
        joined_url = str(urljoin(page_url, href))

        # Fragments would be useful if we chunked a page based on section.
        # Then, each chunk would have a different URL based on the fragment.
        # Since we aren't doing that yet, they just "break" links. So, drop
        # the fragment.
        if drop_fragments:
            return urldefrag(joined_url).url
        return joined_url

    @staticmethod
    def _parse_urls(
        soup: BeautifulSoup, page_url: str, drop_fragments: bool = True
    ) -> list[str]:
        soup_links: list[Tag] = soup.find_all("a")
        urls: set[str] = set()

        for link in soup_links:
            parsed_url = HyperlinkTransformer._parse_url(
                link, page_url=page_url, drop_fragments=drop_fragments
            )
            # Remove self links and entries for any 'a' tag that failed to parse
            # (didn't have href, or invalid domain, etc.)
            if parsed_url and parsed_url != page_url:
                urls.add(parsed_url)

        return list(urls)

    @override
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        results: list[Document] = []
        for document in documents:
            if self._url_metadata_key not in document.metadata:
                msg = (
                    f"html document url not found in metadata at "
                    f"{self._url_metadata_key} for document id: {document.id}"
                )
                raise ValueError(msg)

            page_url = document.metadata[self._url_metadata_key]
            if self._drop_fragments:
                page_url = urldefrag(page_url).url

            soup = BeautifulSoup(document.page_content, "html.parser", **kwargs)
            parsed_urls = self._parse_urls(
                soup=soup, page_url=page_url, drop_fragments=self._drop_fragments
            )

            results.append(
                Document(
                    id=document.id,
                    page_content=document.page_content,
                    metadata={self._metadata_key: parsed_urls, **document.metadata},
                )
            )
        return results
