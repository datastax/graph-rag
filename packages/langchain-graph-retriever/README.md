# LangChain Graph Retriever

LangChain Graph Retriever is a Python library that supports traversing a document graph on top of vector-based similarity search.
It works seamlessly with LangChain's retriever framework and supports various graph traversal strategies for efficient document discovery.

## Features

- **Vector Search**: Perform similarity searches using vector embeddings.
- **Graph Traversal**: Apply traversal strategies such as breadth-first (Eager) or Maximal Marginal Relevance (MMR) to explore document relationships.
- **Customizable Strategies**: Easily extend and configure traversal strategies to meet your specific use case.
- **Multiple Adapters**: Support for various vector stores, including AstraDB, Cassandra, Chroma, OpenSearch, and in-memory storage.
- **Synchronous and Asynchronous Retrieval**: Supports both sync and async workflows for flexibility in different applications.

## Installation

Install the library via pip:

```bash
pip install langchain-graph-retriever
```

## Getting Started

Here is an example of how to use LangChain Graph Retriever:

```python
from langchain_graph_retriever import GraphRetriever
from langchain_core.vectorstores import Chroma

# Initialize the vector store (Chroma in this example)
vector_store = Chroma(embedding_function=your_embedding_function)

# Create the Graph Retriever
retriever = GraphRetriever(
    store=vector_store,
    # Define edges based on document metadata
    edges=[("keywords", "keywords")],
)

# Perform a retrieval
documents = retriever.retrieve("What is the capital of France?")

# Print the results
for doc in documents:
    print(doc.page_content)
```

## License

This project is licensed under the Apache 2 License. See the LICENSE file for more details.
