---
i:
    y: :material-check-circle:{.green}
    n: :material-close-circle:{.red}
    m: :material-alert-circle:{.yellow}
---

# Adapters

Adapters allow `graph-retriever` to connect to specific vector stores.

| Vector Store                   | Supported | Collections | Dict-In-List | Nested Metadata | Optimized Adjacency |
| ------------------------------ | :-------: | :---------: | :----------: | :-------------: | :-----------------: |
| [DataStax Astra](#astra)       | {{ i.y }} | {{ i.y }}   | {{ i.y }}    | {{ i.y }}       | {{ i.y }}           |
| [OpenSearch](#opensearch)      | {{ i.y }} | {{ i.y }}   | {{ i.n }}    | {{ i.n }}       | {{ i.n }}           |
| [Apache Cassandra](#cassandra) | {{ i.y }} | {{ i.m }}   | {{ i.y }}    | {{ i.n }}       | {{ i.n }}           |
| [Chroma](#chroma)              | {{ i.y }} | {{ i.m }}   | {{ i.y }}    | {{ i.n }}       | {{ i.n }}           |

__Supported__

: Indicates whether a given store is completely supported (:material-check-circle:{.green}) or has limited support (:material-alert-circle:{.yellow}).

__Collections__

: Indicates whether the store supports lists in metadata values or not. Stores which do not support it directly (:material-alert-circle:{.yellow}) can be used by applying the [ShreddingTransformer][langchain_graph_retriever.transformers.ShreddingTransformer] document transformer to documents before writing, which spreads the items of the collection into multiple metadata keys.

__Dict-In-List__

: Indicates the store supports using a dict-value in a list for edges. For
example, when using named-entity recognition, you may have `entities = [{"type":
"PERSON", "entity": "Bell"}, ...]` and wish to link nodes with the same entity
using an edge defined as `("entities", "entities")`.

__Nested Metadata__

: Whether edges can be defined using values of nested metadata. For example,
`page_structure.section` to access the section ID stored in metadata as
`metadata["page_structure"] = { "section": ... }`.

__Optimized Adjacency__

: Whether the store supports an optimized query for nodes adjacent to multiple edges. Without this optimization each edge must be queried separately. Stores that support the combined adjacent query perform much better, especially when retrieving large numbers of nodes and/or dealing with high connectivity.

!!! warning

    Graph Retriever can be used with any of these supported Vector Stores. However, stores
    that operate directly on nested collections (without denormalization) and support optimized adjacency
    much more performant and better suited for production use. Stores like Chroma are best
    employed for early experimentation, while it is generally recommended to use a store like DataStax AstraDB when scaling up.

## Supported Stores

### Astra

[DataStax AstraDB](https://www.datastax.com/products/datastax-astra) is
supported by the
[`AstraAdapter`][langchain_graph_retriever.adapters.astra.AstraAdapter]. The adapter
supports operating on metadata containing both primitive and list values.
Additionally, it optimizes the request for nodes connected to multiple edges into a single query.

### OpenSearch

[OpenSearch](https://opensearch.org/) is supported by the [`OpenSearchAdapter`][langchain_graph_retriever.adapters.open_search.OpenSearchAdapter]. The adapter supports operating on metadata containing both primitive and list values. It does not perform an optimized adjacent query.

### Apache Cassandra {: #cassandra}

[Apache Cassandra](https://cassandra.apache.org/) is supported by the [`CassandraAdapter`][langchain_graph_retriever.adapters.cassandra.CassandraAdapter]. The adapter requires shredding metadata containing lists in order to use them as edges. It does not perform an optimized adjacent query.

### Chroma

[Chroma](https://www.trychroma.com/) is supported by the [`ChromaAdapter`][langchain_graph_retriever.adapters.chroma.ChromaAdapter]. The adapter requires shredding metadata containing lists in order to use them as edges. It does not perform an optimized adjacent query.

## Implementation

The [Adapter][graph_retriever.adapters.Adapter] interface may be implemented directly. For LangChain [VectorStores][langchain_core.vectorstores.base.VectorStore], [LangchainAdapter][langchain_graph_retriever.adapters.langchain.LangchainAdapter] and [ShreddedLangchainAdapter][langchain_graph_retriever.adapters.langchain.ShreddedLangchainAdapter] provide much of the necessary functionality.
