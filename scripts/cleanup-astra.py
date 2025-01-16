import os

import astrapy
from astrapy.authentication import StaticTokenProvider

db = astrapy.Database(
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token=StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"]),
    namespace=os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace"),
)
collections = db.list_collection_names()
num_collections = len(collections)

for i, collection in enumerate(collections):
    print(f"Dropping {collection} ({i}/{num_collections})")  # noqa: T201
    db.drop_collection(collection)
