import os
import astrapy
from astrapy.authentication import StaticTokenProvider
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore


db = astrapy.Database(
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"]),
    namespace=os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace"),
)
collections = db.list_collection_names()
num_collections = len(collections)

for i, collection in enumerate(collections):
    print(f"Dropping {collection} ({i}/{num_collections})")
    db.drop_collection(collection)