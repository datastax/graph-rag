# FAQs

## Is Graph RAG a Knowledge Graph?

Yes, for the recent usage of the term.
Graph RAG implements graph traversal of structured metadata during retrieval.
The structured metadata provides edges connecting unstructured content ("knowledge").
Graph RAG traverses a graph of knowledge.

However, prior to the recent surge of Graph RAG, there was a more academic definition of knowledge graphs where nodes specifically represented entities and knowledge about the relationships appeared as edges. Graph RAG is *not* this version of a knowledge graph.

We have found that adding edges to unstructured content is much easier and efficient to use.
See [the Lazy Graph RAG example](../examples/lazy-graph-rag.ipynb) for more details.

??? note "Links with more information."

    We previously wrote about this distinction as ["content-centric" (nodes are content) vs. "entity-centric" (nodes are entities)](https://www.datastax.com/blog/better-llm-integration-and-relevancy-with-content-centric-knowledge-graphs).

    We've also demonstrated that [building the content-centric knowledge graph is significinatly cheaper](https://hackernoon.com/how-to-save-$70k-building-a-knowledge-graph-for-rag-on-6m-wikipedia-pages).

    In many ways this mirrors the difference between Microsoft's GraphRAG and LazyGraphRAG.

## Does Graph RAG need a Graph DB?

No.

Graph databases are excellent for operating on academic knowledge graphs, where you way be looking for specific relationships between multiple nodes -- eg., finding people who live in Seattle (have a "lives in" edge pointing at Seattle) and work at a company in Santa Clara (has a "works at" edge to a company node with a "headquartered in" edge pointing at Santa Clara). In this case, the graph *structure* encodes information, meaning the graph *query* needs to understand that structure.

However, the best knowledge graph for Graph RAG is a vector store containing unstructured content with structured metadata first, and support traversal of those structured relationships second. This means that any vector store with metadata filtering capabilities (all or nearly all) can be used for traversal.

!!! important

    Traditional graph databases require materializing edges during ingestion, making them inflexible and costly to maintain as data evolves. Our approach operates on relationships present in the metadata without materializing them, eliminating the need to decide on the graph relationships during ingestion and enabling each query to operate on a different set of relationships. This makes it easy to add your structured metadata to the documents and traverse it for enhanced retrieval in RAG applications and adapts effortlessly to changing data.

There are some things a vector store can support that make the kinds of metadata queries needed for traversal more efficient. See the support matrix in the [Adapters guide](../guide/adapters.md) for more information.