# 📘 GraphRAG Interaction Guide

## Overview

GraphRAG is a retrieval-augmented generation (RAG) framework that combines unstructured vector similarity with structured graph traversal over document metadata. Each document is a node, and metadata fields define directed edges between them. This enables nuanced and context-aware retrieval tailored to complex reasoning tasks.

---

## 🧠 Graph Structure and Metadata

Each document is represented as a node in a graph. Edges between nodes are defined dynamically via shared or directional metadata fields.

> ⚠️ The metadata fields shown below are examples. A complete list of supported fields and their descriptions will be published as a separate resource.

### Defining Edges

Edges are declared as tuples: `(source_field, destination_field)`.

```python
edges = [
    ("keywords", "keywords"),         # Connect by shared keyword
    ("authors", "authors"),           # Connect documents by the same author
    ("cites", "$id"),                 # Document A cites another by ID
    ("$id", "cited_by"),              # Documents that cite this one
]
```

### Edge Directionality

Each edge has a direction: from source ➡ to destination. This matters when building and traversing the graph.

#### Special token: `$id`

- `$id` refers to the document’s unique ID.
- It allows edges to reference specific documents by ID in metadata.

**Example Usages:**
- `("references", "$id")`: Walk from documents that reference others to those they reference.
- `("$id", "references")`: Inverse — walk to documents that reference this one.

---

## 🔍 Retrieval Strategies

GraphRAG supports retrieval through graph traversal and semantic similarity. The retrieval behavior is controlled via traversal depth and strategy.

### Depth-0 Retrieval (Standard Retriever Mode)

If `max_depth = 0`, GraphRAG behaves as a standard vector-based retriever. No graph traversal is performed.

```python
retriever = GraphRetriever(
    store=vector_store,
    edges=[],
    strategy="eager",
    max_depth=0,
    select_k=5,
)
```

### Traversal from Vector Search Results

Set `start_k > 0` to perform initial vector search. The resulting documents serve as traversal roots.

### Traversal from Known Roots (No Vector Search)

Set `start_k = 0` and provide `initial_roots` (list of document IDs). This disables vector search and starts traversal from known nodes.

```python
retriever = GraphRetriever(
    store=vector_store,
    edges=[("references", "$id")],
    strategy="mmr",
    initial_roots=["doc-123", "doc-456"],
    select_k=10,
    max_depth=2
)
```

---

## 🧠 Strategy Types

### Eager Strategy

- Selects all reachable nodes during traversal.
- Prioritizes completeness.

### MMR (Max Marginal Relevance) Strategy

- Selects contextually relevant but diverse documents at each traversal step.
- `lambda_mult` adjusts tradeoff between relevance (1.0) and diversity (0.0).
- `min_mmr_score` filters out nodes with low marginal score.

---

## ⚙ Configuration Parameters

| Parameter         | Type                    | Description                                                                                     | Default     |
|-------------------|-------------------------|-------------------------------------------------------------------------------------------------|-------------|
| `query`           | `str`                   | Natural language query used for initial vector search. Ignored if `start_k = 0`.                | `""`        |
| `edges`           | `List[Tuple[str, str]]` | Metadata fields used to form graph edges.                                                       | `[]`        |
| `strategy`        | `str`                   | Traversal strategy: `"eager"` or `"mmr"`.                                                       | `"eager"`   |
| `select_k`        | `int`                   | Number of documents to return. Equivalent to `k` in most retrievers.                            | `10`        |
| `start_k`         | `int`                   | Number of vector search results to use as traversal roots. Set to `0` to disable vector search. | `5`         |
| `initial_roots`   | `List[str]`             | Document IDs to start traversal from. Used only if traversal is enabled.                        | `[]`        |
| `adjacent_k`      | `int`                   | Max number of adjacent nodes considered per traversal step.                                     | `5`         |
| `max_depth`       | `int`                   | Maximum traversal depth. `0` disables traversal (vector-only behavior).                         | `1`         |
| `lambda_mult`     | `float`                 | (MMR only) Trade-off between relevance (1.0) and diversity (0.0).                               | `0.5`       |
| `min_mmr_score`   | `float`                 | (MMR only) Filter: only include nodes with MMR score ≥ this threshold.                          | `0.0`       |
| `metadata_filter` | `dict[str, Any]`        | Optional: restrict traversal to nodes whose metadata matches this filter.                       | `None`      |

---

## 📎 Example: MMR Traversal from Known Roots

```python
retriever = GraphRetriever(
    store=vector_store,
    edges=[("cites", "$id"), ("authors", "authors")],
    strategy="mmr",
    initial_roots=["doc-001", "doc-005"],
    select_k=8,
    adjacent_k=3,
    max_depth=2,
    lambda_mult=0.6
)
```

In this example, retrieval begins from the two specified documents, traversing out via shared authorship and citation. No vector search is performed.

