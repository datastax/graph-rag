import os

from astrapy import AstraDBDatabaseAdmin
from astrapy.authentication import StaticTokenProvider

api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]
token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")

if keyspace != "default_keyspace":
    admin = AstraDBDatabaseAdmin(
        api_endpoint=api_endpoint,
        token=token,
    )
    admin.drop_keyspace(keyspace)
