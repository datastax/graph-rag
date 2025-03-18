from textwrap import indent

from langchain_core.documents import Document


def _add_tabs(text: str) -> str:
    return indent(text, "\t")


def _format_parameter(el: dict[str, str]) -> str:
    text = el["name"]
    if "value" in el and "default" in el:
        assert el["value"] == el["default"]

    if "type" in el:
        text += f": {el['type']}"
    if "default" in el:
        text += f" = {el['default']}"
    if "description" in el:
        desc = _add_tabs(el["description"])
        text += f"\n\t{desc}"
    return text


def _format_return(el: dict[str, str]) -> str:
    items = []
    if "type" in el:
        items.append(el["type"])
    if "description" in el:
        items.append(_add_tabs(el["description"]))
    return "\n\t".join(items)


def format_document(doc: Document, debug: bool = False) -> str:
    """Format a document as documentation for including as context in a LLM query."""
    metadata = doc.metadata
    text = f"{metadata['name']} ({metadata['kind']})\n\n"

    text += f"path: \n\t{metadata['path']}\n\n"

    for key in ["bases", "exports", "implemented_by"]:
        if key in metadata:
            values = "\n".join(metadata[key])
            text += f"{key}: \n\t{_add_tabs(values)}\n\n"

    if "properties" in metadata:
        props = [f"{k}: {v}" for k, v in metadata["properties"].items()]
        values = "\n".join(props)
        text += f"properties: \n\t{_add_tabs(values)}\n\n"

    if doc.page_content != "":
        text += f"description: \n\t{_add_tabs(doc.page_content)}\n\n"
    elif "value" in metadata:
        text += f"{metadata['value']}\n\n"

    for key in ["attributes", "parameters"]:
        if key in metadata:
            values = "\n\n".join([_format_parameter(v) for v in metadata[key]])
            text += f"{key}: \n\t{_add_tabs(values)}\n\n"

    for key in ["returns", "yields"]:
        if key in metadata:
            values = "\n\n".join([_format_return(v) for v in metadata[key]])
            text += f"{key}: \n\t{_add_tabs(values)}\n\n"

    for key in ["note", "example"]:
        if key in metadata:
            text += f"{key}: \n\t{_add_tabs(metadata[key])}\n\n"

    if debug:
        if "imports" in metadata:
            imports = []
            for as_name, real_name in metadata["imports"].items():
                if real_name == as_name:
                    imports.append(real_name)
                else:
                    imports.append(f"{real_name} as {as_name}")
            values = "\n".join(imports)
            text += f"imports: \n\t{_add_tabs(values)}\n\n"

        for key in ["references", "gathered_types"]:
            if key in metadata:
                values = "\n".join(metadata[key])
                text += f"{key}: \n\t{_add_tabs(values)}\n\n"

        if "parent" in metadata:
            text += f"parent: {metadata['parent']}\n\n"

    return text


def format_docs(docs: list[Document]) -> str:
    """Format documents as documentation for including as context in a LLM query."""
    return "\n---\n".join(format_document(doc) for doc in docs)
