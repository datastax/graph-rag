import json
from typing import Any

file = "docs/_inv/langchain_objects.json"

langchain_class_remapping = {
    'langchain_astradb.vectorstores.AstraDBVectorStore': 'langchain_astradb.AstraDBVectorStore',
    'langchain_chroma.vectorstores.Chroma': 'langchain_chroma.Chroma',
    'langchain_core.documents.base.Document': 'langchain_core.documents.Document',
    'langchain_core.vectorstores.in_memory.InMemoryVectorStore': 'langchain_core.vectorstores.InMemoryVectorStore',
    'langchain_core.vectorstores.base.VectorStore': 'langchain_core.vectorstores.VectorStore',
    'langchain_community.vectorstores.opensearch_vector_search.OpenSearchVectorSearch': 'langchain_community.vectorstores.OpenSearchVectorSearch',
    'langchain_community.vectorstores.cassandra.Cassandra': 'langchain_community.vectorstores.Cassandra'
}

objects: dict[str, Any] = {}
with open(file, "r") as f:
    objects = json.load(f)

items: list[dict[str, Any]] = objects["items"]

new_items = []

for item in items:
    if item["name"] in langchain_class_remapping.keys():
        new_item = item.copy()
        new_item["name"] = langchain_class_remapping[item["name"]]
        new_items.append(new_item)

items.extend(new_items)

with open(file, "w") as f:
    json.dump(obj=objects, fp=f)
