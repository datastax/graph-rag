import os

from astrapy import AstraDBDatabaseAdmin
from astrapy.authentication import StaticTokenProvider
import httpx

api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]
token = StaticTokenProvider(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
keyspace = os.environ.get("ASTRA_DB_KEYSPACE", "default_keyspace")

if keyspace != "default_keyspace":
    admin = AstraDBDatabaseAdmin(
        api_endpoint=api_endpoint,
        token=token,
    )
    if keyspace in admin.list_keyspaces():
        for _i in range(3):
            try:
                admin.drop_keyspace(keyspace)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 409:
                    raise e
