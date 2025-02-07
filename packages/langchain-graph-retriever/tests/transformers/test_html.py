import pytest
from langchain_core.documents import Document

test_html = """
<!DOCTYPE html>
<html><head><title>Animals of the World</title></head>
<body>
    <h1>Animals of the World</h1>
    <p>Explore the fascinating world of animals.</p>

    <h2>Mammals</h2>
    <p>The <a href="https://example.com/lion">lion</a> is the king of the jungle.</p>
    <p>The <a href="https://example.com/elephant">elephant</a> is a large animal.</p>

    <h2>Birds</h2>
    <p>The <a href="https://example.com/eagle">eagle</a> soars high in the sky.</p>
    <p>The <a href="https://example.com/penguin">penguin</a> thrives in icy areas.</p>

    <h2>Reptiles</h2>
    <p>The <a href="https://example.com/crocodile">crocodile</a> is a predator.</p>
    <p>The <a href="https://example.com/gecko">gecko</a> can climb walls.</p>
</body></html>
"""


@pytest.mark.extra
def test_transform_documents():
    from langchain_graph_retriever.transformers.html import (
        HyperlinkTransformer,
    )

    doc = Document(
        id="animal_html",
        page_content=test_html,
        metadata={"_url": "https://example.com/animals"},
    )

    original_doc = doc.model_copy()

    transformer = HyperlinkTransformer(
        url_metadata_key="_url",
        metadata_key="_hyperlinks",
    )

    transformed_doc = transformer.transform_documents([doc])[0]
    assert "_hyperlinks" in transformed_doc.metadata
    assert "https://example.com/gecko" in transformed_doc.metadata["_hyperlinks"]
    assert len(transformed_doc.metadata["_hyperlinks"]) == 6

    transformer = HyperlinkTransformer()
    with pytest.raises(ValueError, match="html document url not found in metadata"):
        transformer.transform_documents([doc])

    # confirm original docs aren't modified
    assert original_doc == doc
