from .base import Adapter
from langchain_core.vectorstores import VectorStore
import importlib

ADAPTERS_PKG = "langchain_graph_retriever.adapters"
_KNOWN_STORES = {
    "langchain_astradb.AstraDBVectorStore": (f"{ADAPTERS_PKG}.astra", "AstraAdapter"),
    "langchain_community.vectorstores.Cassandra": (f"{ADAPTERS_PKG}.cassandra", "CassandraAdapter"),
    "langchain_chroma.Chroma": (f"{ADAPTERS_PKG}.chroma", "ChromaAdapter"),
    "langchain_core.vectorstores.in_memory.InMemoryVectorStore": (f"{ADAPTERS_PKG}.in_memory", "InMemoryAdapter"),
    "langchain_community.vectorstores.OpenSearchVectorSearch": (f"{ADAPTERS_PKG}.open_search", "OpenSearchAdapter"),
}

# Class names that indicate we don't need to keep traversing.
STOP_NAMES = {
    "abc.ABC",
    "builtins.object",
    "langchain_core.vectorstores.base.VectorStore",
}

def infer_adapter(store: Adapter | VectorStore) -> Adapter:
    """Infer the adapter to use for the given store."""
    if isinstance(store, Adapter):
        return store

    store_classes = [store.__class__]
    while store_classes:
        store_class = store_classes.pop()

        store_name = f"{store_class.__module__}.{store_class.__name__}"
        if store_name in STOP_NAMES:
            continue

        adapter = _KNOWN_STORES.get(store_name, None)
        if adapter is not None:
            module_name, class_name = adapter
            adapter_module = importlib.import_module(module_name)
            adapter_class = getattr(adapter_module, class_name)
            return adapter_class(store)

        # If we didn't find it yet, and the naem wasn't a stopping point,
        # we queue up the base classes for consideration. This allows
        # matching subclasses of supported vector stores.
        store_classes.extend(store_class.__bases__)

    return None
