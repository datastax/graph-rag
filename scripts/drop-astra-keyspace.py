import os

from astrapy import AstraDBDatabaseAdmin
from astrapy.authentication import StaticTokenProvider

token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")
api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

admin = AstraDBDatabaseAdmin(api_endpoint=api_endpoint, token=token)
if keyspace in admin.list_keyspaces():
    admin.drop_keyspace(keyspace)
