from collections.abc import Iterable

from graph_retriever import Node


def sorted_doc_ids(nodes: Iterable[Node]) -> list[str]:
    return sorted([n.id for n in nodes if n.id is not None])
