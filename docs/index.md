---
hide:
  - navigation
  - toc
  - path
---

# Graph RAG

Graph RAG provides retrievers combining **unstructured** similarity-search on vectors and
**structured** traversal of metadata properties.
These retrievers are implemented using the metadata search functionality of existing vector stores, **allowing you to traverse your existing vector store**!

<div class="grid cards two" markdown>

-   :material-transit-connection-variant:{ .lg .middle } __Link based on existing metadata__

    ---

    Use existing metadata fields without additional processing.
    Retrieve more from your existing vector store!

    [:octicons-arrow-right-24: Get started](./get-started/index.md)

-   :material-clock-edit-outline:{ .lg .middle } __Change links on demand__

    ---

    Edges can be specified on-the-fly, allowing different relationships to be traversed based on the question.

    [:octicons-arrow-right-24: Edges](./get-started/edges.md)


-   :material-connection:{ .lg .middle } __Pluggable Traversal Strategies__

    ---

    Use built-in traversal strategies like Eager or MMR, or define your own logic to select which nodes to explore.

    [:octicons-arrow-right-24: Strategies](./get-started/strategies.md)

-   :material-multicast:{ .lg .middle } __Broad compatibility__

    ---

    Adapters are available for a variety of vector stores with support for
    additional stores easily added.

    [:octicons-arrow-right-24: Adapters](./get-started/adapters.md)
</div>

## Example: LangChain Retriever combining Vector and Graph traversal

```python
from langchain_graph_retriever import GraphRetriever
from graph_retriever.edges import Id

retriever = GraphRetriever(
    store = store,
    edges = [("mentions", Id()), ("entities", "entities")], # (1)!
)

retriever.invoke("where is Santa Clara?")
```

1. `edges` configures traversing from a node to other nodes listed in the `metadata["mentions"]` field (to the corresponding `id`) and to other nodes with overlapping `metadata["entities"]`.

See [Examples](examples/index.md) for more complete examples.