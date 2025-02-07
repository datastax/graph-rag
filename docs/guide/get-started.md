# Get Started

This page demonstrates how to combine Graph Traversal and Vector Search using `langchain-graph-retriever` with `langchain`.

## Pre-requisites

We assume you already have a working `langchain` installation, including an LLM and embedding model as well as a [supported vector store](./adapters.md).

In that case, you only need to install `langchain-graph-retriever`:

```bash
pip install langchain langchain-graph-retriever
```

## Preparing Data

Loading data is exactly the same as for whichever vector store you use.
The main thing to consider is what structured information you wish to include in the metadata to support traversal.

For this guide, I have a JSON file with information about animals. Several example entries are shown below. The actual file has one entry per line, making it easy to load into `Document`s.

```json
{
    "id": "alpaca",
    "text": "alpacas are domesticated mammals valued for their soft wool and friendly demeanor.",
    "metadata": {
        "type": "mammal",
        "number_of_legs": 4,
        "keywords": ["wool", "domesticated", "friendly"],
        "origin": "south america"
    }
}
{
    "id": "caribou",
    "text": "caribou, also known as reindeer, are migratory mammals found in arctic regions.",
    "metadata": {
        "type": "mammal",
        "number_of_legs": 4,
        "keywords": ["migratory", "arctic", "herbivore", "tundra"],
        "diet": "herbivorous"
    }
}
{
    "id": "cassowary",
    "text": "cassowaries are flightless birds known for their colorful necks and powerful legs.",
    "metadata": {
        "type": "bird",
        "number_of_legs": 2,
        "keywords": ["flightless", "colorful", "powerful"],
        "habitat": "rainforest"
    }
}
```

```python title="Fetching Animal Data"
from graph_rag_example_helpers.datasets.animals import fetch_documents
animals = fetch_documents()
```

## Populating the Vector Store

The following shows how to populate a variety of vector stores with the animal data.

=== "Astra"

    ```python
    from dotenv import load_dotenv
    from langchain_astradb import AstraDBVectorStore
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()
    vector_store = AstraDBVectorStore.from_documents(
        collection_name="animals",
        documents=animals,
        embedding=OpenAIEmbeddings(),
    )
    ```

=== "Apache Cassandra"

    ```python
    from langchain_community.vectorstores.cassandra import Cassandra
    from langchain_openai import OpenAIEmbeddings
    from langchain_graph_retriever.transformers import ShreddingTransformer

    shredder = ShreddingTransformer() # (1)!
    vector_store = Cassandra.from_documents(
        documents=list(shredder.transform_documents(animals)),
        embedding=OpenAIEmbeddings(),
        table_name="animals",
    )
    ```

    1. Since Cassandra doesn't index items in lists for querying, it is necessary to
    shred metadata containing list to be queried. By default, the
    [ShreddingTransformer][langchain_graph_retriever.transformers.ShreddingTransformer]
    shreds all keys. It may be configured to only shred those
    metadata keys used as edge targets.

=== "OpenSearch"

    ```python
    from langchain_community.vectorstores import OpenSearchVectorSearch
    from langchain_openai import OpenAIEmbeddings

    vector_store = OpenSearchVectorSearch.from_documents(
        opensearch_url=OPEN_SEARCH_URL,
        index_name="animals",
        embedding_function=OpenAIEmbeddings(),
        engine="faiss",
        documents=animals,
    )
    ```

=== "Chroma"

    ```python
    from langchain_chroma.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_graph_retriever.transformers import ShreddingTransformer

    shredder = ShreddingTransformer() # (1)!
    vector_store = Chroma.from_documents(
        documents=list(shredder.transform_documents(animals)),
        embedding=OpenAIEmbeddings(),
        collection_name_name="animals",
    )
    ```

    1. Since Chroma doesn't index items in lists for querying, it is necessary to
    shred metadata containing list to be queried. By default, the
    [ShreddingTransformer][langchain_graph_retriever.transformers.ShreddingTransformer]
    shreds all keys. It may be configured to only shred those
    metadata keys used as edge targets.

## Simple Traversal

For our first retrieval and graph traversal, we're going to start with a single animal best matching the query, and then traverse to other animals with the same `habitat` and/or `origin`.

=== "Astra"

    ```python
    from graph_retriever.strategies import Eager
    from langchain_graph_retriever import GraphRetriever

    simple = GraphRetriever(
        store = vector_store,
        edges = [("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
        strategy = Eager(k=10, start_k=1, depth=2),
    )
    ```

=== "Apache Cassandra"

    ```python
    from graph_retriever.strategies import Eager
    from langchain_graph_retriever import GraphRetriever
    from langchain_graph_retriever.adapters.cassandra import CassandraAdapter

    simple = GraphRetriever(
        store = store = CassandraAdapter(vector_store, shredder, {"keywords"}),,
        edges = [("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
        strategy = Eager(k=10, start_k=1, depth=2),
    )
    ```

=== "OpenSearch"

    ```python
    from graph_retriever.strategies import Eager
    from langchain_graph_retriever import GraphRetriever

    simple = GraphRetriever(
        store = vector_store,
        edges = [("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
        strategy = Eager(k=10, start_k=1, depth=2),
    )
    ```


=== "Chroma"

    ```python
    from graph_retriever.strategies import Eager
    from langchain_graph_retriever import GraphRetriever
    from langchain_graph_retriever.adapters.chroma import ChromaAdapter

    simple = GraphRetriever(
        store = ChromaAdapter(vector_store, shredder, {"keywords"}),
        edges = [("habitat", "habitat"), ("origin", "origin"), ("keywords", "keywords")],
        strategy = Eager(k=10, start_k=1, depth=2),
    )
    ```

!!! note "Shredding"

    The above code is exactly the same for all stores, however adapters for shredded stores (Chroma and Apache Cassandra) require configuration to specify which metadata fields need to be rewritten when issuing queries.

The above creates a graph traversing retriever that starts with the nearest animal (`start_k=1`), retrieves 10 documents (`k=10`) and limits the search to documents that are at most 2 steps away from the first animal (`depth=2`).

The edges define how metadata values can be used for traversal. In this case, every animal is connected to other animals with the same habitat and/or same origin.

```python
simple_results = simple.invoke("what mammals could be found near a capybara")

for doc in simple_results:
    print(f"{doc.id}: {doc.page_content}")
```

## Visualizing

`langchain-graph-retrievers` includes code for converting the document graph into a `networkx` graph, for rendering and other analysis.
See @fig-document-graph

```python title="Graph retrieved documents"
import networkx as nx
import matplotlib.pyplot as plt
from langchain_graph_retriever.document_graph import create_graph

document_graph = create_graph(
    documents=simple_results,
    edges = simple.edges,
)

nx.draw(document_graph, with_labels=True)
plt.show()
```