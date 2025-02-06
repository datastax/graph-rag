---
title: "Introducing Graph Retrievers: Smarter, Simpler Document Graphs for Vector Stores"
sug: introducing-graph-rag
description: "Announcing the first release of Graph Retrievers, a powerful graph traversal retriever for your vector store!"
author: "Ben Chambers"
date: 2025-01-31
categories:
    - langchain
    - news
---

We're excited to announce the release of **Graph Retrievers**, a powerful new tool for leveraging graph traversal in your vector stores with ease!

With Graph Retrievers, you can dynamically explore relationships between documents using metadata fields—no need for complex preprocessing or building an entire knowledge graph upfront.

<!-- more -->

## A Brief History: Where We Started

We originally developed [`GraphVectorStore`](https://www.datastax.com/blog/knowledge-graphs-for-rag-without-a-graphdb) to efficiently handle structured relationships between documents. This approach proved especially useful for [reducing costs in knowledge graph creation](https://hackernoon.com/how-to-save-$70k-building-a-knowledge-graph-for-rag-on-6m-wikipedia-pages). By lazily traversing metadata instead of building a full graph, we made real-time retrieval more efficient and cost-effective.

Recently, [Microsoft introduced LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) which found similar cost and performance benefits by linking documents based on named entities rather than building a full knowledge graph.

Since GraphVectorStore was introduced into LangChain, the concept of traversing document graphs has evolved significantly, and today, Graph Retrievers offers an easier and more flexible way to bring graph-like capabilities to your vector stores.

## What’s New in Graph Retrievers?

1. **Effortless Metadata Linking**
   Documents can now be linked via metadata fields without additional processing. You define relationships on-the-fly.
   Use different configurations to tailor traversal to your needs, such as exploring citations or co-authorships.

2. **Pluggable Traversal Strategies**
   In addition to built-in strategies like eager traversal and MMR, you can now define your own logic for graph exploration.

3. **Broad Compatibility**
   Adapters are available for DataStax Astra DB, Apache Cassandra, Chroma DB, and OpenSearch, with support for additional stores easily added.

## Example: Getting Started with Graph Retrievers

Here’s how you can use Graph Retrievers with an existing `AstraDBVectorStore` that includes metadata fields for article mentions and named entities.

Assuming you already have a LangChain project using a Vector Store, all you need to do is:

1. The following example assumes you already have a LangChain Vector Store.
   We're using an existing `AstraDBVectorStore` similar to:

    ```python
    from langchain_astradb import AstraDBVectorStore
    from langchain_openai import OpenAIEmbeddings

    vector_store = AstraDBVectorStore(
        collection_name="animals",
        embedding=OpenAIEmbeddings(),
    )
    ```

2. Install `langchain-graph-retriever`:

    ```sh
    pip install langchain-graph-retriever
    ```

3. Add the following code using your existing Vector Store.

    ```python
    from langchain_graph_retriever import GraphRetriever

    # Define your graph traversal
    traversal = GraphRetriever(
        store=vector_store,
        edges=[("mentions", "id"), ("entities", "entites")],
    )

    # Query the graph
    traversal.invoke("Where is Lithuania?")
    ```

With just a few lines of code, you can navigate relationships between articles, dynamically retrieving the most relevant information for your query.

## Try It Out Today!
Reflecting these improvements, we've moved the implementation to a new [package](https://pypi.org/project/langchain-graph-retriever/) and [repository](https://github.com/datastax/graph-rag), making it even easier to integrate and explore.

- **Documentation**: Learn how to get started in the [official documentation](https://datastax.github.io/graph-rag).
- **Join the Community**: Share feedback or contribute by opening an issue or pull request in the [GitHub repo](https://github.com/datastax/graph-rag).

Give Graph Retrievers a try today and take your retrieval-augmented generation (RAG) workflows to the next level. We can’t wait to hear what you build!
