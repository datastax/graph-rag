# Examples

<div class="grid cards" markdown>
-   :material-code-braces-box:{ .lg .middle } __Lazy Graph RAG__

    ---

    Implements [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) using LangChain and `langchain-graph-retriever`.

    It loads Wikipedia articles and traverses based on links ("mentions") and named entities (extracted from the content). It retrieves a large number of articles, groups them by community, and extracts claims from each community. The best claims are used to answer the question.

    [:material-fast-forward: Lazy Graph RAG Example](lazy-graph-rag.ipynb)

-   :material-code-braces-box:{ .lg .middle } __Code Generation__

    ---
    This example notebook shows how to load documentation for python packages into a
    vector store so that it can be used to provide context to an LLM for code generation.

    It uses LangChain and `langchain-graph-retriever` with a custom traversal Strategy
    in order to improve LLM generated code output. It shows that using GraphRAG can
    provide a significant increase in quality over using either an LLM alone or standard
    RAG.

    GraphRAG traverses cross references in the documentation like a software engineer
    would, in order to determine how to solve a coding problem.

    [:material-fast-forward: Code Generation Example](code-generation.ipynb)
</div>
