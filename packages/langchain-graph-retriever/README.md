# LangChain Graph Retriever

LangChain Graph Retriever is a Python library that integrates vector-based similarity search with graph traversal strategies to retrieve documents. It is designed to work seamlessly with LangChain's retriever framework and supports various graph traversal strategies for efficient document discovery.

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

### Example Usage

```python
from langchain_graph_retriever import GraphRetriever
from langchain_core.vectorstores import Chroma

# Initialize the vector store (Chroma in this example)
vector_store = Chroma(embedding_function=your_embedding_function)

# Define edge relationships and traversal strategy
edges = ["keywords"]  # Define edges based on document metadata
strategy = Eager(k=10)  # Use breadth-first traversal

# Create the Graph Retriever
retriever = GraphRetriever(store=vector_store, edges=edges, strategy=strategy)

# Perform a retrieval
query = "What is the capital of France?"
documents = retriever.retrieve(query)

# Print the results
for doc in documents:
    print(doc.page_content)
```

## Components

### Adapters
Adapters bridge the gap between vector stores and the graph retriever. The following adapters are supported:

- **AstraAdapter**: Integrates with AstraDB.
- **CassandraAdapter**: Integrates with Cassandra.
- **ChromaAdapter**: Integrates with Chroma.
- **OpenSearchAdapter**: Integrates with OpenSearch.
- **InMemoryAdapter**: For lightweight, in-memory use cases.

### Strategies
Strategies define the traversal behavior:

- **Eager**: A breadth-first traversal strategy.
- **Scored**: Uses a scoring function to prioritize nodes.
- **MMR**: Implements Maximal Marginal Relevance for balancing relevance and diversity.

### Utilities
- **EdgeHelper**: Extracts and encodes edges from document metadata.
- **DocumentGraph**: Creates and analyzes document graphs.

## Advanced Configuration

### Custom Strategy
You can implement your own traversal strategy by extending the `Strategy` base class:

```python
from langchain_graph_retriever.strategies import Strategy

class CustomStrategy(Strategy):
    def score(self, document, **kwargs):
        # Define custom scoring logic
        return 1.0

    def select_nodes(self, *, limit):
        # Define custom node selection logic
        return []

    def finalize_nodes(self, nodes):
        # Define custom finalization logic
        return nodes

    def discover_nodes(self, nodes):
        # Define custom discovery logic
        pass
```

### Integration with Other Vector Stores
If you are using a custom or unsupported vector store, you can implement an adapter by extending the `Adapter` base class.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## License

This project is licensed under the Apache 2 License. See the LICENSE file for more details.
