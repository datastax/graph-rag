# Adapters

Adapters allow `graph-retriever` to connect to specific vector stores.

## Implementation
The [Adapter][graph_retriever.adapters.Adapter] interface may be implemented directly. For LangChain [VectorStores][langchain_core.vectorstores.base.VectorStore], [LangchainAdapter][langchain_graph_retriever.adapters.langchain.LangchainAdapter] and [DenormalizedAdapter][langchain_graph_retriever.adapters.langchain.DenormalizedAdapter] provide much of the necessary functionality.