# Transformers

!!! note "Transformers are optional, not mandatory"
    Graph traversal operates on the structured metadata.
    Transformers provide tools for populating the metadata, but they are not necessary.
    In many cases you may have existing structured information that is useful
    in addition or instead of what the transformers would populate.

We provide two types of document transformers that can be useful in setting up your
documents for graph traversal.

* **Information Extractors:** These extract information out of document content
    and add to the metadata.

* **Metadata Utilities:** These add to or modify document metadata to enable certain
    features

## Information Extractors

!!! note "Extras required"
    Most of the Transformers in this section require extra packages to be installed.
    Either look at the specifics in the reference documentation for each transformer,
    or install all the extras via:

    ```
    pip install "langchain-graph-retriever[all]"
    ```

### NLP-Model Based

Several of our document transformers that extract information depend on pre-trained
Natural Language Processing (NLP) models.

The following LangChain documents will be used for the code examples in this section:

??? example "Test Documents"
    ```python
    from langchain_core.documents import Document

    model_docs = [
        Document(
            id="red_fox",
            page_content="""
    The Red Fox is an omnivore, feeding on small mammals, birds, fruits, and insects. It
    thrives in a wide range of habitats, including forests, grasslands, and even urban areas
    like New York City, where it has adapted to human presence. This agile creature moves
    primarily by walking and running, but it can also leap and climb when necessary. Its
    body is covered in thick fur, which helps it stay warm in colder climates. The National
    Wildlife Federation has tracked their urban expansion, and their population was
    highlighted in the Wildlife Conservation Summit 2023.""",
        ),
        Document(
            id="sea_turtle",
            page_content="""
    The Green Sea Turtle is a herbivore, grazing on seagrass and algae in coastal waters and
    shallow tropical seas, particularly around the Great Barrier Reef. It is a powerful
    swimmer, using its large, flipper-like limbs to glide through the ocean. Unlike mammals,
    its body is covered in a tough, scaly shell, providing protection from predators.
    Conservation efforts by The World Wildlife Fund have played a significant role in
    protecting this species, and it was a major focus of discussion at the Marine Life
    Protection Conference 2024.",
        ),
    ]
    ```

#### GLiNERTransformer

The [`GLiNERTransformer`][langchain_graph_retriever.transformers.gliner.GLiNERTransformer]
extracts structured entity labels from text, identifying key attributes and categories
to enrich document metadata with semantic information.

Example use:
    ```python
    from pprint import pprint
    from langchain_graph_retriever.transformers.gliner import GLiNERTransformer
    gliner = GLiNERTransformer(labels=["diet", "habitat", "locomotion", "body covering"])

    gliner_docs = gliner.transform_documents(docs)
    for doc in gliner_docs:
        pprint({"id": doc.id, "metadata": doc.metadata}, width=100)
    ```

Example output:
    ```text
    {'id': 'red_fox',
    'metadata': {'body covering': ['thick fur'],
                'diet': ['birds', 'omnivore', 'small mammals', 'insects', 'fruits'],
                'habitat': ['urban areas', 'new york city', 'forests', 'grasslands'],
                'locomotion': ['walking and running']}}
    {'id': 'sea_turtle',
    'metadata': {'body covering': ['scaly shell'],
                'diet': ['seagrass and algae'],
                'habitat': ['coastal waters', 'shallow tropical seas', 'great barrier reef']}}
    ```

#### KeyBERTTransformer

The [`KeyBERTTransformer`][langchain_graph_retriever.transformers.keybert.KeyBERTTransformer]
extracts key topics and concepts from text, generating metadata that highlights the most
relevant terms to describe the content.

Example use:
    ```python
    from langchain_graph_retriever.transformers.keybert import KeyBERTTransformer
    keybert = KeyBERTTransformer()

    keybert_docs = keybert.transform_documents(model_docs)
    for doc in keybert_docs:
        print(f"{doc.id}: {doc.metadata}")
    ```

Example output:
    ```text
    red_fox: {'keywords': ['wildlife', 'fox', 'mammals', 'habitats', 'omnivore']}
    sea_turtle: {'keywords': ['turtle', 'reef', 'marine', 'seagrass', 'wildlife']}
    ```

#### SpacyNERTransformer

The [`SpacyNERTransformer`][langchain_graph_retriever.transformers.spacy.SpacyNERTransformer]
identifies and labels named entities in text, extracting structured metadata such as organizations, locations, dates, and other key entity types.

Example use:
    ```python
    from pprint import pprint
    from langchain_graph_retriever.transformers.spacy import SpacyNERTransformer
    spacy = SpacyNERTransformer()

    spacy_docs = spacy.transform_documents(docs)
    for doc in spacy_docs:
        pprint({"id": doc.id, "metadata": doc.metadata}, width=100)
    ```

Example output:
    ```text
    {'id': 'red_fox',
    'metadata': {'entities': ['ORG: The National Wildlife Federation',
                            'GPE: New York City',
                            'ORG: the Wildlife Conservation Summit',
                            'DATE: 2023']}}
    {'id': 'sea_turtle',
    'metadata': {'entities': ['ORG: The World Wildlife Fund',
                            'FAC: the Great Barrier Reef',
                            'ORG: the Marine Life Protection Conference',
                            'LOC: The Green Sea Turtle',
                            'DATE: 2024']}}
    ```

### Parser Based

The following document transformer uses a parser to extract metadata.

#### HyperlinkTransformer

The [`HyperlinkTransformer`][langchain_graph_retriever.transformers.html.HyperlinkTransformer]
extracts hyperlinks from HTML content and stores them in document metadata.

??? example "Test Html Documents"
    ```python
    from langchain_core.documents import Document
    animal_html = """
        <!DOCTYPE html>
        <html><head><title>Animals of the World</title></head>
        <body>
            <h2>Mammals</h2>
            <p>The <a href="https://example.com/lion">lion</a> is the king of the jungle.</p>
            <p>The <a href="https://example.com/elephant">elephant</a> is a large animal.</p>

            <h2>Birds</h2>
            <p>The <a href="https://example.com/eagle">eagle</a> soars high in the sky.</p>
            <p>The <a href="https://example.com/penguin">penguin</a> thrives in icy areas.</p>
        </body></html>
        """

    html_doc = Document(
        page_content=animal_html,
        metadata={"url": "https://example.com/animals"}
    )
    ```

    Note that each document needs to have an existing `url` metadata field.

Example use:
    ```python
    from pprint import pprint
    from langchain_graph_retriever.transformers.html import HyperlinkTransformer
    html_transformer = HyperlinkTransformer()

    extracted_doc = html_transformer.transform_documents(html_docs)[0]

    pprint(extracted_doc.metadata)
    ```

Example output:
    ```text
    {'hyperlink': ['https://example.com/eagle',
                'https://example.com/lion',
                'https://example.com/elephant',
                'https://example.com/penguin'],
    'url': 'https://example.com/animals'}
    ```

## Metadata Utilities

### ParentTransformer

The [`ParentTransformer`][langchain_graph_retriever.transformers.ParentTransformer]
adds the hierarchal `Parent` path to the document metadata.

??? example "Test Documents"
    ```python
    from langchain_core.documents import Document

    parent_docs = [
        Document(id="root", page_content="test", metadata={"path": "root"}),
        Document(id="h1", page_content="test", metadata={"path": "root.h1"}),
        Document(id="h1a", page_content="test", metadata={"path": "root.h1.a"}),
    ]
    ```

    Note that each document needs to have an existing `path` metadata field.

Example use:
    ```python
    from langchain_graph_retriever.transformers import ParentTransformer
    transformer = ParentTransformer(path_delimiter=".")

    transformed_docs = transformer.transform_documents(parent_docs)
    for doc in transformed_docs:
        print(f"{doc.id}: {doc.metadata}")
    ```

Example output:
    ```text
    root: {'path': 'root'}
    h1: {'path': 'root.h1', 'parent': 'root'}
    h1a: {'path': 'root.h1.a', 'parent': 'root.h1'}
    ```

### ShreddingTransformer

The [`ShreddingTransformer`][langchain_graph_retriever.transformers.ShreddingTransformer]
is primarily designed as a helper utility for vector stores that do not have native
support for collection-based metadata fields. It transforms these fields into multiple
metadata key-value pairs before database insertion. It also provides a method to restore
metadata back to its original format.

#### Shredding

??? example "Test Document"
    ```python
    from langchain_core.documents import Document

    collection_doc = Document(id="red_fox", page_content="test", metadata={
        "diet": ["birds", "omnivore", "small mammals", "insects", "fruits"],
        "size": "small"
    })
    ```

Example use:
    ```python
    from pprint import pprint
    from langchain_graph_retriever.transformers import ShreddingTransformer

    shredder = ShreddingTransformer()
    shredded_docs = shredder.transform_documents([collection_doc])
    pprint(shredded_docs[0].metadata)
    ```

Example output:
    ```text
    {'__shredded_keys': '["diet"]',
    'diet→birds': '§',
    'diet→fruits': '§',
    'diet→insects': '§',
    'diet→omnivore': '§',
    'diet→small mammals': '§',
    'size': 'small'}
    ```

#### Restoration

This example uses the output from the Shredding Example above.

Example use:
    ```python
    restored_docs = shredder.restore_documents(shredded_docs)
    pprint(restored_docs[0].metadata)
    ```

Example output:
    ```text
    {'diet': ['birds', 'omnivore', 'small mammals', 'insects', 'fruits'],
     'size': 'small'}
    ```