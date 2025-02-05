"""Infers the appropriate adapter for a given vector store."""

import importlib

from graph_retriever.adapters import Adapter
from langchain_core.vectorstores.base import VectorStore

ADAPTERS_PKG = "langchain_graph_retriever.adapters"
_KNOWN_STORES = {
    "langchain_astradb.vectorstores.AstraDBVectorStore": (
        f"{ADAPTERS_PKG}.astra",
        "AstraAdapter",
    ),
    "langchain_community.vectorstores.cassandra.Cassandra": (
        f"{ADAPTERS_PKG}.cassandra",
        "CassandraAdapter",
    ),
    "langchain_chroma.vectorstores.Chroma": (f"{ADAPTERS_PKG}.chroma", "ChromaAdapter"),
    "langchain_core.vectorstores.in_memory.InMemoryVectorStore": (
        f"{ADAPTERS_PKG}.in_memory",
        "InMemoryAdapter",
    ),
    (
        "langchain_community.vectorstores."
        "opensearch_vector_search.OpenSearchVectorSearch"
    ): (
        f"{ADAPTERS_PKG}.open_search",
        "OpenSearchAdapter",
    ),
}

# Class names that indicate we don't need to keep traversing.
STOP_NAMES = {
    "abc.ABC",
    "builtins.object",
    "langchain_core.vectorstores.base.VectorStore",
}


def _full_class_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _infer_adapter_name(cls: type) -> tuple[str, str]:
    """Return the module and class of the adapter or raise."""
    store_classes = [cls]
    while store_classes:
        store_class = store_classes.pop()

        store_class_name = _full_class_name(store_class)
        if store_class_name in STOP_NAMES:
            continue

        adapter = _KNOWN_STORES.get(store_class_name, None)
        if adapter is not None:
            return adapter

        # If we didn't find it yet, and the naem wasn't a stopping point,
        # we queue up the base classes for consideration. This allows
        # matching subclasses of supported vector stores.
        store_classes.extend(store_class.__bases__)

    store_class_name = _full_class_name(cls)
    raise ValueError(
        f"Expected adapter or supported vector store, but got {store_class_name}"
    )


def infer_adapter(store: Adapter | VectorStore) -> Adapter:
    """
    Dynamically infer the adapter for a given vector store.

    This function identifies the correct adapter based on the vector store type
    and instantiates it with the provided arguments.

    Parameters
    ----------
    store :
        The vector store instance.

    Returns
    -------
    :
        The initialized adapter for the given vector store.
    """
    if isinstance(store, Adapter):
        return store

    module_name, class_name = _infer_adapter_name(store.__class__)
    adapter_module = importlib.import_module(module_name)
    adapter_class = getattr(adapter_module, class_name)
    return adapter_class(store)
