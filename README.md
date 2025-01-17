# Graph RAG

Retrievers providing both unstructured (similarity-search on vectors) and
structured (traversal of metadata properties).


<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/datastax/graph-rag">
    <img src="https://github.com/datastax/graph-rag/raw/main/images/logo.webp" alt="Logo" width="160" height="160">
  </a>

<h3 align="center">Graph RAG - Combining Vector and Graph for better RAG</h3>

  <p align="center">
    Library providing Graph RAG combining vector search and traversal of metadata relationships.
    <br />
    <a href="https://github.com/datastax/graph-rag"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <a href="https://github.com/datastax/graph-rag/issues">Report Bug</a>
    ·
    <a href="https://github.com/datastax/graph-rag/issues">Request Feature</a>
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

Graph RAG provides retrievers combining vector-search (for unstructured similarity) and traversal (for structured relationships in metadata).
These retrievers are implemented using the metadata search functionality of existing vector stores, **allowing you to traverse your existing vector store**!

The core library (`graph-retriever`) supports can be used in generic Python applications, while `langchain-graph-retriever` provides [langchain](https://python.langchain.com/docs/introduction/)-specific functionality.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started with LangChain

1. Install `langchain-graph-retriever` (or add to your Python dependencies).

    ```sh
    pip install langchain-graph-retriever
    ```

1. Wrap your existing vector store to enable graph retrieval:

    ```python
    from langchain_graph_retriever.retrievers import GraphTraversalRetriever
    from langchain_graph_retriever.retrievers.strategy import Mmr
    from langchain_graph_retriever.retrievers.store_adapters.astra import AstraStoreAdapter

    retriever = GraphTraversalRetriever(
        # Adapt AstraDBVectorStore for use with Graph Retrievers.
        # Exposes functionality of the underlying store that is otherwise not available.
        store = AstraStoreAdapter(store),
        # Define the relationships to navigate:
        #   1. From nodes with a list of `mentions` to the nodes with the corresponding `ids`.
        #   2. From noeds with a list of related `entities` to other nodes with the same entities.
        edges = [("mentions", "id"), "entities"],
        # The "strategy" to use for traversing the graph. In this case, max-marginal relevance.
        strategy = Mmr(k=10),
    )

    retriever.invoke("where is Santa Clara?")
    ```

## Roadmap

Graph RAG is under active development.
This is an overview of our current roadmap - please 👍 issues that are important to you.
Don't see a feature that would be helpful for your application - [create a feature request](https://github.com/datastax/graph-rag/issues)!

* Support more vector stores
* Support [Lazy Graph RAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) on the retrieved
  documents.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](`CONTRIBUTING.md`) for more information on development.

<p align="right">(<a href="#readme-top">back to top</a>)</p

<!-- LICENSE -->
## License

Distributed under the Apache 2 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>