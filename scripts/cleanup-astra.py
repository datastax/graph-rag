import os

import astrapy
from astrapy.authentication import StaticTokenProvider

keyspace=os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
keyspace = keyspace.replace("/", "_")
db = astrapy.Database(
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"]),
    keyspace=keyspace,
)
collections = db.list_collection_names()
num_collections = len(collections)

for i, collection in enumerate(collections):
    print(f"Dropping {collection} ({i}/{num_collections})")  # noqa: T201
    db.drop_collection(collection)
