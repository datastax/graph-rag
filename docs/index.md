# Graph RAG

Graph RAG provides retrievers combining **unstructured** similarity-search on vectors and
**structured** traversal of metadata properties.
These retrievers are implemented using the metadata search functionality of existing vector stores, **allowing you to traverse your existing vector store**!

The core library, `graph-retriever`, can be used in generic Python applications. For applications using [LangChain](https://python.langchain.com/docs/introduction/), `langchain-graph-retriever` provides a `Retriever` and integrations with existing vector stores.

## Example

The following is a short example demonstrating how to create a LangChain
[Retriever][langchain_core.retrievers.BaseRetriever] which will retrieve content
using both vector search and relationships between `metadata["mentions"]` and
`id`, as well as between `metadata["entities"]`.

```
from langchain_graph_retriever import GraphRetriever
from graph_retriever.edges import Id

retriever = GraphRetriever(
    store = store,
    # Define the relationships to navigate:
    #   1. From `metadata["mentions"]` to nodes with the corresponding `id`.
    #   2. From `metadata["entities"]` to nodes with the same entities.
    edges = [("mentions", Id()), ("entities", "entities")],
)

retriever.invoke("where is Santa Clara?")
```

See [Examples](examples/index.md) for more complete examples.