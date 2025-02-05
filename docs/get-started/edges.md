# Edges

Edges specify how content should be linked.
Often, content in existing vector stores has metadata based on structured information.
For example, a vector store containing articles may have information about the authors, keywords, and citations of those articles.
__Such content can be traversed along relationships already present in that metadata!__
See [Specifying Edges](#specifying-edges) for more on how edges are specified.

## Specifying Edges {: #specifying-edges}

```python title="Example content"
Content(
    id="article1",
    content="...",
    metadata={
        "keywords": ["GPT", "GenAI"],
        "authors": ["Ben", "Eric"],
        "primary_author": "Eric",
        "cites": ["article2", "article3"],
    }
)
```

1. `("keywords", "keywords")` connects to other articles about GPT and GenAI.
2. `("authors", "authors")` connects to other articles by any of the same authors.
3. `("authors", "primary_author")` connects to other articles whose primary author was Ben or Eric.
4. `("cites", Id())` connects to the articles cited (by ID).
5. `(Id(), "cites")` connects to articles which cite this one.
6. `("cites", "cites")` connects to other articles with citations in common.

## Edge Functions

While sometimes the information to traverse is missing and the vector store
needs to be re-populated, in other cases the information exist but not quite be
in a suitable format for traversal. For instance, the `"authors"` field may
contain a list of authors and their institution, making it impossible to link to
other articles by the same author when they were at a different institution.

In such cases, you can provide a custom
[`EdgeFunction`][graph_retriever.edges.EdgeFunction] to extract the edges for
traversal.