from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urldefrag, urljoin, urlparse

from langchain_core._api import beta
from langchain_core.documents import BaseDocumentTransformer, Document

if TYPE_CHECKING:
    from bs4 import BeautifulSoup  # type: ignore
    from bs4.element import Tag  # type: ignore


@beta()
class HtmlHyperlinkExtractor(BaseDocumentTransformer):
    """Extract hyperlinks from HTML content.

    Expects each document to contain its URL in its metadata.

    Example::

        extractor = HtmlHyperlinkExtractor()
        results = extractor.extract_one(HtmlInput(html, url))

    .. seealso::

        - :mod:`How to use a graph vector store <langchain_community.graph_vectorstores>`
        - :class:`How to create links between documents <langchain_community.graph_vectorstores.links.Link>`

    How to link Documents on hyperlinks in HTML
    ===========================================

    Preliminaries
    -------------

    Install the ``beautifulsoup4`` package:

    .. code-block:: bash

        pip install -q langchain_community beautifulsoup4

    Usage
    -----

    For this example, we'll scrape 2 HTML pages that have an hyperlink from one
    page to the other using an ``AsyncHtmlLoader``.
    Then we use the ``HtmlLinkExtractor`` to create the links in the documents.

    Using extract_one()
    ^^^^^^^^^^^^^^^^^^^

    We can use :meth:`extract_one` on a document to get the links and add the links
    to the document metadata with
    :meth:`~langchain_community.graph_vectorstores.links.add_links`::

        from langchain_community.document_loaders import AsyncHtmlLoader
        from langchain_community.graph_vectorstores.extractors import (
            HtmlInput,
            HtmlLinkExtractor,
        )
        from langchain_community.graph_vectorstores.links import add_links
        from langchain_core.documents import Document

        loader = AsyncHtmlLoader(
            [
                "https://python.langchain.com/docs/integrations/providers/astradb/",
                "https://docs.datastax.com/en/astra/home/astra.html",
            ]
        )

        documents = loader.load()

        html_extractor = HtmlLinkExtractor()

        for doc in documents:
            links = html_extractor.extract_one(HtmlInput(doc.page_content, url))
            add_links(doc, links)

        documents[0].metadata["links"][:5]

    .. code-block:: output

        [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

    Using as_document_extractor()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    If you use a document loader that returns the raw HTML and that sets the source
    key in the document metadata such as ``AsyncHtmlLoader``,
    you can simplify by using :meth:`as_document_extractor` that takes directly a
    ``Document`` as input::

        from langchain_community.document_loaders import AsyncHtmlLoader
        from langchain_community.graph_vectorstores.extractors import HtmlLinkExtractor
        from langchain_community.graph_vectorstores.links import add_links

        loader = AsyncHtmlLoader(
            [
                "https://python.langchain.com/docs/integrations/providers/astradb/",
                "https://docs.datastax.com/en/astra/home/astra.html",
            ]
        )
        documents = loader.load()
        html_extractor = HtmlLinkExtractor().as_document_extractor()

        for document in documents:
            links = html_extractor.extract_one(document)
            add_links(document, links)

        documents[0].metadata["links"][:5]

    .. code-block:: output

        [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

    Using LinkExtractorTransformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Using the :class:`~langchain_community.graph_vectorstores.extractors.link_extractor_transformer.LinkExtractorTransformer`,
    we can simplify the link extraction::

        from langchain_community.document_loaders import AsyncHtmlLoader
        from langchain_community.graph_vectorstores.extractors import (
            HtmlLinkExtractor,
            LinkExtractorTransformer,
        )
        from langchain_community.graph_vectorstores.links import add_links

        loader = AsyncHtmlLoader(
            [
                "https://python.langchain.com/docs/integrations/providers/astradb/",
                "https://docs.datastax.com/en/astra/home/astra.html",
            ]
        )

        documents = loader.load()
        transformer = LinkExtractorTransformer(
            [HtmlLinkExtractor().as_document_extractor()]
        )
        documents = transformer.transform_documents(documents)

        documents[0].metadata["links"][:5]

    .. code-block:: output

        [Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/spreedly/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/nvidia/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/ray_serve/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/integrations/providers/bageldb/'),
            Link(kind='hyperlink', direction='out', tag='https://python.langchain.com/docs/introduction/')]

    We can check that there is a link from the first document to the second::

        for doc_to in documents:
            for link_to in doc_to.metadata["links"]:
                if link_to.direction == "in":
                    for doc_from in documents:
                        for link_from in doc_from.metadata["links"]:
                            if (
                                link_to.direction == "in"
                                and link_from.direction == "out"
                                and link_to.tag == link_from.tag
                            ):
                                print(
                                    f"Found link from {doc_from.metadata['source']} to {doc_to.metadata['source']}."
                                )

    .. code-block:: output

        Found link from https://python.langchain.com/docs/integrations/providers/astradb/ to https://docs.datastax.com/en/astra/home/astra.html.

    The documents with URL links can then be added to a :class:`~langchain_community.graph_vectorstores.base.GraphVectorStore`::

        from langchain_community.graph_vectorstores import CassandraGraphVectorStore

        store = CassandraGraphVectorStore.from_documents(documents=documents, embedding=...)

    Args:
        kind: The kind of edge to extract. Defaults to ``hyperlink``.
        drop_fragments: Whether fragments in URLs and links should be
            dropped. Defaults to ``True``.

    """  # noqa: E501

    def __init__(
        self,
        *,
        url_metadata_key: str = "url",
        metadata_key: str = "hyperlink",
        drop_fragments: bool = True,
    ):
        try:
            from bs4 import BeautifulSoup  # noqa:F401
        except ImportError as e:
            raise ImportError(
                "BeautifulSoup4 is required for HtmlHyperlinkExtractor. "
                "Please install it with `pip install beautifulsoup4`."
            ) from e

        self._url_metadata_key = url_metadata_key
        self._metadata_key = metadata_key
        self._drop_fragments = drop_fragments

    @staticmethod
    def _parse_url(link: Tag, page_url: str, drop_fragments: bool = True) -> str | None:
        href = link.get("href")
        if href is None:
            return None
        url = urlparse(href)
        if url.scheme not in ["http", "https", ""]:
            return None

        # Join the HREF with the page_url to convert relative paths to absolute.
        url = str(urljoin(page_url, href))

        # Fragments would be useful if we chunked a page based on section.
        # Then, each chunk would have a different URL based on the fragment.
        # Since we aren't doing that yet, they just "break" links. So, drop
        # the fragment.
        if drop_fragments:
            return urldefrag(url).url
        return url

    @staticmethod
    def _parse_urls(
        soup: BeautifulSoup, page_url: str, drop_fragments: bool = True
    ) -> list[str]:
        soup_links: list[Tag] = soup.find_all("a")
        urls: set[str] = set()

        for link in soup_links:
            parsed_url = HtmlHyperlinkExtractor._parse_url(
                link, page_url=page_url, drop_fragments=drop_fragments
            )
            # Remove self links and entries for any 'a' tag that failed to parse
            # (didn't have href, or invalid domain, etc.)
            if parsed_url and parsed_url != page_url:
                urls.add(parsed_url)

        return list(urls)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Extract hyperlinks from html documents using BeautifulSoup4.

        Args:
            documents: The documents to transform.
            kwargs: Arguments to pass through to the BeautifulSoup parser.

        """
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

            document.metadata[self._metadata_key] = self._parse_urls(
                soup=soup, page_url=page_url, drop_fragments=self._drop_fragments
            )
        return documents
