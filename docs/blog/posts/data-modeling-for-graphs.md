---
title: "Data Modeling for Graph Retrieval"
slug: data-modeling-for-graphs
description: "Best-practices for structuring metadata to support Graph Retrieval."
author: "Ben Chambers"
date: 2025-02-21
categories:
    - data modeling
---

Retrieval Augmented Generation (RAG) – using vector similarity search to
retrieve semantically relevant information – is great. It is easy to implement
and provides great results for many questions. However, as great as it is there
are cases where it breaks down and advanced RAG techniques may be needed:
* Complex questions needing information about multiple topics may require question rewriting or an agent based approach.
* Broad questions may need multiple perspectives, and benefit from techniques like MMR to retrieve a diverse set of results.
* Deep questions may require explanations and clarifying information for the retrieved context, benefitting from knowledge graphs and/or agent techniques asking follow-up questions.

Knowledge graphs linking between content solve these problems. Graph RAG lets
us start with vector similarity to address a wide variety of questions. As
vector similarity breaks down, links between documents can be traversed to
retrieve deeper information and handle more complex questions. Graph traversal
techniques like MMR can be used to retrieve a diverse set of contexts also helps
with broad questions.

By enabling these graph traversals using metadata in the vector store, Graph RAG allows you to solve these problems without re-ingesting your data. At retrieval time, documents may be linked to other documents based on properties stored in the metadata.

In this article, we’ll see how easy it is to use an existing vector store as a graph vector store. We'll understand the kind of information that may be useful to have in the metadata to support these traversals.

## What to link
When first upgrading a vector store to a GraphVectorStore, the question of what to link naturally arises. Since vector similarity does such a great job of capturing semantic similarity based on the content the best use of links is to capture relationships that may not be readily apparent from the content alone. For example:

1. In a legal document you could create links from paragraphs to definitions of relevant terms.
1. In a technical support knowledge base you could link to more specific steps for each suggested procedure.
1. In an article, you could link a paragraph of text to an image that illustrates the concept.
1. In a travel application you could link a hotel to nearby attractions in the same metropolitan area.

In all of these cases, the links provide additional information that is not semantically related to the original content. They are the kinds of things that you might link to if you were writing a webpage or look for in a follow-up search when performing deeper research. Often, this information is available in your existing API or database and just needs to be added to the retrieval system.

## Example: IMDB API

We previously showed using movie descriptions from IMDB as documents when
constructing a knowledge graph. In that example, we extracted keywords and other
properties to use as links. We’re going to revisit that example, but using the
IMDB API to demonstrate creating links from the existing structured information.

For our use case, we’re focused on retrieving information about movies, so we’ll
use bidirectional links and only create a LangChain `Document` for each movie.

We're going to put information about 

```python
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_community.graph_vectorstores import Link
from imdb.Movie import Movie

def persons_to_links(movie: Movie, field: str) -> Iterable[Link]:
  for person in movie[field]:
    if person.personID is not None:
      yield Link.bidir(field, person["name"])

def movie_to_document(movie_id: str) -> Document:
  movie = ia.get_movie(movie_id)

  links = []
  # As bidirectional links, these allow navigating to other movies with the same
  # link. We could instead use an outgoing link to director, and an incoming
  # link with information about the director, if we wanted to represent
  # directors explicitly as points to visit during retrieval.
  links.extend(persons_to_links(movie, "directors"))
  links.extend(persons_to_links(movie, "cinematographer"))
  links.append(Link.bidir("year", str(movie["year"])))
  links.extend([Link.bidir("genre", g) for g in movie["genres"]])
  links.extend(persons_to_links(movie, "writer"))

  document = Document(id = movie.movieID,
                      page_content=f'{movie["title"]}\PLOT: {movie.get("plot outline") or movie.get("plot")}',
                      metadata={ 
                        "directors": persons(movie, "directors"),
                        "cinematographer": persons(movie, "cinematagropher"),
                        "year": str(movie["year"]),
                        "genres": movie["genres"],
                        "writer": persons(movie, "writer"),
                        })
  return document
```

## Conclusion

Documents contain _unstructured_ content. Document metadata captures
_structured_ information about that content. Edges connect documents based on
the metadata.

When populating your vector store, make sure the information you already have is
available on the metadata. Adding this information allows you to use metadata
filtering and graph traversal to navigate these dimensions in ways not captured
by semantic similarity on the content.
