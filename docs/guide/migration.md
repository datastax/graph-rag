# Migration

This page discusses migration from LangChain `GraphVectorStore` as well as between versions of `graph-retriever` and `langchain-graph-retriever`.

## From LangChain GraphVectorStore

LangChain `GraphVectorStore` relied on putting specially crafted `Link` instances into `metadata["links"]`. Many cases used link extractors to compute these links, but it was also often useful (and necessary) to create them manually.

When converting from a `GraphVectorStore` to the new `langchain-graph-retriever` library, you need to do the following:

1. Replace uses of the link extractors with document transformers.
2. Replace manualy link creation with metadata fields.
3. Replace `GraphVectorStore` usage with the `GraphRetriever`.

### Replace Link Extractors with Document Transformers

`GLiNERLinkExtractor`

: Replace with [GLiNERTransformer][langchain_graph_retriever.transformers.gliner.GLiNERTransformer], which will populate metadata fields for each label.

`HierarchyLinkExtractor`

: If you already have a parent ID in the metadata, you can remove this. Otherwise, replace with the [ParentTransformer][langchain_graph_retriever.transformers.ParentTransformer] which populates a `parent` field computed from a path. The parent field may be used with edges to achieve parent-to-child, child-to-parent, and sibling-to-sibling navigation.

`HtmlLinkExtractor`

: Replace with [HyperlinkTransformer][langchain_graph_retriever.transformers.html.HyperlinkTransformer] which extracts hyperlinks from each chunk and writes them to a metadata field.

`KeybertLinkExtractor`

: Replace with [KeybertTransformer][langchain_graph_retriever.transformers.keybert.KeyBERTTransformer], which will populate a metadata field with the keywords.

### Replace Manual Link Creation with Metadata Fields

With the old library, you had to choose the direction of the links when they were created -- either `in`, `out` or `bidir`. With the new library, you simply create the corresponding fields and choose the direction of edges when you issue a query (see [next section](#replace-graphvectorstore-with-the-graphretriever)).

```py title="GraphVectorStore Links (Old)"
# Document metadata for a page at `http://somesite` linking to some other URLs
# and a few keyword links.
doc = Document(
    ...,
    metadata = {
        "links": [
            Link.incoming("url", "http://somesite"),
            Link.outgoing("url", "http://someothersite"),
            Link.outgoing("url", "http://yetanothersite"),
            Link.bidir("keyword", "sites"),
            Link.bidir("keyword", "web"),
        ]
    }
)

```

```py title="LangChain Graph Retriever (New)"
doc = Document(
    ...,
    metadata = {
        "url": "http://somesite",
        "hrefs": ["http://someothersite", "http://yetanothersite"],
        "keywords": ["sites", "web"],
    }
)
```

These metadata fields can be used to accomplish a variety of graph traversals. For example:

* `edges = [("hrefs", "url"), ...]` navigates from a site to the pages it links to (from `hrefs` to `url`).
* `edges = [("keywords", "keywords"), ...]` navigates from a site to other sites with the same keyword.
* `edges = [("url", "hrefs"), ...]` navigates from a site to other sites that link to it.

!!! tip "Per-Query Edges"

    You can use different edges for each query, allowing you to navigate different directions depending on the needs. In the old library, you only ever navigated out from a site to the things it linked to, while with the new library the metadata captures the information (what URL is this document from, what URLs does it reference) and the edges determine which fileds are traversed at retrieval time.

### Replace GraphVectorStore with the GraphRetriever

Finally, rather than creating the links and writing them to a `GraphVectorStore` you write the documents (with metadata) to a standard `VectorStore` and apply a [GraphRetriever][langchain_graph_retriever.GraphRetriever]:

```py title="LangChain Graph Retriever (New)"
from langchain_graph_retriever import GraphRetriever
retriever = GraphRetriever(
    store=vector_store,
    edges=[("hrefs", "url"), ("keywords", "keywords")],
)
```