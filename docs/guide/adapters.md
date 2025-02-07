# Adapters

Adapters allow `graph-retriever` to connect to specific vector stores.

| Vector Store                   | Supported                       | Collection Support               | Combined Adjacent Query         |
| ------------------------------ | | | |
| [DataStax Astra](#astra)       | :material-check-circle:{.green} | :material-check-circle:{.green}  | :material-check-circle:{.green} |
| [OpenSearch](#opensearch)      | :material-check-circle:{.green} | :material-check-circle:{.green}  |                                 |
| [Apache Cassandra](#cassandra) | :material-check-circle:{.green} | :material-alert-circle:{.yellow} |                                 |
| [Chroma](#chroma)              | :material-check-circle:{.green} | :material-alert-circle:{.yellow} |                                 |

__Supported__

: Indicates whether a given store is completely supported (:material-check-circle:{.green}) or has limited support (:material-alert-circle:{.yellow}).

__Collection Support__

: Indicates whether the store supports lists in metadata values or not. Stores which do not support it directly (:material-alert-circle:{.yellow}) can be used by applying the [ShreddingTransformer][langchain_graph_retriever.transformers.ShreddingTransformer] document transformer to documents before writing, which spreads the items of the collection into multiple metadata keys.

__Combined Adjacent Query__

: Whether the store supports the more efficient "combined adjacent query" to retrieve nodes adjacent to multiple edges in a single query. Stores which don't use the combined query instead use a fallback implementation which issues a query for each edge. Stores that support the combined adjacent query perform much better, especially when retrieving large numbers of nodes and/or dealing with high connectivity.

!!! warning

    Graph Retriever can be used with any of these supported Vector Stores. However, stores
    that operate directly on normalized data and perform the combined adjacent query are
    much more performant and better suited for production use. Stores like Chroma are best
    employed for early experimentation, while it is generally recommended to use a store like DataStax AstraDB when scaling up.

## Supported Stores

### Astra

[DataStax AstraDB](https://www.datastax.com/products/datastax-astra) is
supported by the
[`AstraAdapter`][langchain_graph_retriever.adapters.astra.AstraAdapter]. The adapter
supports operating on metadata containing both primitive and list values. Additionally, it optimizes the request for nodes connected to multiple edges into a single query.

### OpenSearch

[OpenSearch](https://opensearch.org/) is supported by the [`OpenSearchAdapter`][langchain_graph_retriever.adapters.open_search.OpenSearchAdapter]. The adapter supports operating on metadata containing both primitive and list values. It does not combine the adjacent query.

### Apache Cassandra {: #cassandra}

[Apache Cassandra](https://cassandra.apache.org/) is supported by the [`CassandraAdapter`][langchain_graph_retriever.adapters.cassandra.CassandraAdapter]. The adapter requires shredding metadata containing lists in order to use them as edges. It does not combine the adjacent query.

### Chroma

[Chroma](https://www.trychroma.com/) is supported by the [`ChromaAdapter`][langchain_graph_retriever.adapters.chroma.ChromaAdapter]. The adapter requires shredding metadata containing lists in order to use them as edges. It does not combine the adjacent query.

## Implementation

The [Adapter][graph_retriever.adapters.Adapter] interface may be implemented directly. For LangChain [VectorStores][langchain_core.vectorstores.base.VectorStore], [LangchainAdapter][langchain_graph_retriever.adapters.langchain.LangchainAdapter] and [ShreddedLangchainAdapter][langchain_graph_retriever.adapters.langchain.ShreddedLangchainAdapter] provide much of the necessary functionality.