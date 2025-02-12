import os
import sys

from astrapy import AstraDBDatabaseAdmin
from astrapy.authentication import StaticTokenProvider

token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
keyspace = os.environ.get("ASTRA_DB_KEYSPACE")
if keyspace is None:
    print("No keyspace to drop.")  # noqa: T201
    sys.exit()

api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

admin = AstraDBDatabaseAdmin(api_endpoint=api_endpoint, token=token)
keyspaces = admin.list_keyspaces()
if keyspace in keyspaces:
    print(f"Dropping keyspace '{keyspace}'")  # noqa: T201
    result = admin.drop_keyspace(keyspace)
    print(f"Dropped keyspace '{keyspace}: {result}")  # noqa: T201
else:
    print(f"Not dropping keyspace '{keyspace}'. Not in {keyspaces}")  # noqa: T201
