from langchain_core.documents import Document

from graph_rag_example_helpers.utils import add_tabs


def _format_parameter(el: dict[str, str]) -> str:
    text = el["name"]
    if "value" in el and "default" in el:
        assert el["value"] == el["default"]

    if "type" in el:
        text += f": {el['type']}"
    if "default" in el:
        text += f" = {el['default']}"
    if "description" in el:
        desc = add_tabs(el["description"])
        text += f"\n\t{desc}"
    return text


def _format_return(el: dict[str, str]) -> str:
    items = []
    if "type" in el:
        items.append(el["type"])
    if "description" in el:
        items.append(add_tabs(el["description"]))
    return "\n\t".join(items)


def format_document(doc: Document, debug: bool = False) -> str:
    """Format a document as documentation for including as context in a LLM query."""
    metadata = doc.metadata
    text = f"{metadata['name']} ({metadata['kind']})\n\n"

    text += f"path: \n\t{metadata['path']}\n\n"

    if "bases" in metadata:
        text += f"bases: \n\t{add_tabs('\n'.join(metadata['bases']))}\n\n"

    if "exports" in metadata:
        text += f"exports: \n\t{add_tabs('\n'.join(metadata['exports']))}\n\n"

    if "implemented_by" in metadata:
        text += (
            f"implemented_by: \n\t{add_tabs('\n'.join(metadata['implemented_by']))}\n\n"
        )

    if "properties" in metadata:
        props = [f"{k}: {v}" for k, v in metadata["properties"].items()]
        text += f"properties: \n\t{add_tabs('\n'.join(props))}\n\n"

    if doc.page_content != "":
        text += f"description: \n\t{add_tabs(doc.page_content)}\n\n"
    elif "value" in metadata:
        text += f"{metadata['value']}\n\n"

    if "attributes" in metadata:
        attributes = [_format_parameter(a) for a in metadata["attributes"]]
        text += f"attributes: \n\t{add_tabs('\n\n'.join(attributes))}\n\n"

    if "parameters" in metadata:
        parameters = [_format_parameter(p) for p in metadata["parameters"]]
        text += f"parameters: \n\t{add_tabs('\n\n'.join(parameters))}\n\n"

    if "returns" in metadata:
        returns = [_format_return(r) for r in metadata["returns"]]
        text += f"returns: \n\t{add_tabs('\n\n'.join(returns))}\n\n"

    if "yields" in metadata:
        yields = [_format_return(y) for y in metadata["yields"]]
        text += f"yields: \n\t{add_tabs('\n\n'.join(yields))}\n\n"

    if "note" in metadata:
        text += f"note: \n\t{add_tabs(metadata['note'])}\n\n"

    if "example" in metadata:
        text += f"example: \n\t{add_tabs(metadata['example'])}\n\n"

    if debug:
        if "imports" in metadata:
            imports = []
            for as_name, real_name in metadata["imports"].items():
                if real_name == as_name:
                    imports.append(real_name)
                else:
                    imports.append(f"{real_name} as {as_name}")
            text += f"imports: \n\t{add_tabs('\n'.join(imports))}\n\n"

        if "references" in metadata:
            text += f"references: \n\t{add_tabs('\n'.join(metadata['references']))}\n\n"

        if "gathered_types" in metadata:
            text += f"gathered_types: \n\t{add_tabs('\n'.join(metadata['gathered_types']))}\n\n"  # noqa: E501

        if "parent" in metadata:
            text += f"parent: {metadata['parent']}\n\n"

    return text


def format_docs(docs: list[Document]) -> str:
    """Format documents as documentation for including as context in a LLM query."""
    return "\n---\n".join(format_document(doc) for doc in docs)
