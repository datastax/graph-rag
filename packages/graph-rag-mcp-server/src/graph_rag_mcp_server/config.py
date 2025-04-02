import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, Optional

from pydantic import Field, model_validator, SecretStr
from pydantic_settings import BaseSettings

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class ServerConfig(BaseSettings):
    # --- Required config ---
    embeddings_type: Literal["openai", "anthropic"] = Field(..., alias="embeddings", env="EMBEDDINGS")
    store_type: Literal["astra", "astradb", "chroma", "opensearch", "cassandra"] = Field(..., alias="store", env="STORE")

    # Astra config
    astra_token: Optional[SecretStr] = Field(None, alias="astra_db_application_token", env="ASTRA_DB_APPLICATION_TOKEN")
    astra_keyspace: Optional[str] = Field("default_keyspace", alias="astra_db_keyspace", env="ASTRA_DB_KEYSPACE")
    astra_endpoint: Optional[str] = Field(None, alias="astra_db_api_endpoint", env="ASTRA_DB_API_ENDPOINT")
    astra_collection: Optional[str] = Field(None, alias="astra_db_collection_name", env="ASTRA_DB_COLLECTION_NAME")

    # Chroma config
    chroma_collection: Optional[str] = Field("default", alias="chroma_collection", env="CHROMA_COLLECTION")
    chroma_persist_directory: Optional[str] = Field(None, alias="chroma_persist_directory", env="CHROMA_PERSIST_DIRECTORY")

    # OpenSearch config
    opensearch_url: Optional[str] = Field(None, alias="opensearch_url", env="OPENSEARCH_URL")
    opensearch_index_name: Optional[str] = Field("default_index", alias="opensearch_index_name", env="OPENSEARCH_INDEX_NAME")

    # Cassandra config
    cassandra_contact_points: Optional[str] = Field(None, alias="cassandra_contact_points", env="CASSANDRA_CONTACT_POINTS")
    cassandra_keyspace: Optional[str] = Field("default_keyspace", alias="cassandra_keyspace", env="CASSANDRA_KEYSPACE")
    cassandra_table_name: Optional[str] = Field("documents", alias="cassandra_table_name", env="CASSANDRA_TABLE_NAME")

    # --- Internal objects ---
    embedding: Optional[Embeddings] = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
        "case_sensitive": False,
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def validate_required_fields_and_embedding(self) -> "ServerConfig":
        logger.debug(f"Validating model — embeddings_type={self.embeddings_type}, store_type={self.store_type}")

        if self.store_type in ("astra", "astradb"):
            missing = [
                k for k in ["astra_token", "astra_endpoint", "astra_collection"]
                if getattr(self, k) is None
            ]
            if missing:
                raise ValueError(f"Missing required Astra config: {', '.join(missing)}")

        elif self.store_type == "opensearch":
            if not self.opensearch_url:
                raise ValueError("Missing required config: opensearch_url")

        elif self.store_type == "cassandra":
            if not self.cassandra_contact_points:
                raise ValueError("Missing required config: cassandra_contact_points")

        # Load embedding based on embeddings_type
        if self.embeddings_type == "openai":
            from langchain_openai.embeddings import OpenAIEmbeddings
            self.embedding = OpenAIEmbeddings()
        elif self.embeddings_type == "anthropic":
            from langchain_anthropic.embeddings import AnthropicEmbeddings
            self.embedding = AnthropicEmbeddings()
        else:
            raise ValueError(f"Unsupported embeddings type: {self.embeddings_type}")

        return self

    def validate_connections(self) -> None:
        logger.debug("Validating embedding model connection...")
        _ = self.embedding.embed_query("test")

        if self.store_type in ("astra", "astradb"):
            from astrapy.authentication import StaticTokenProvider
            from langchain_astradb import AstraDBVectorStore

            store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.astra_collection,
                namespace=self.astra_keyspace,
                token=StaticTokenProvider(self.astra_token.get_secret_value()),
                api_endpoint=self.astra_endpoint,
            )
            _ = store.similarity_search("ping", k=1)

        elif self.store_type == "chroma":
            from langchain_chroma import Chroma

            store = Chroma(
                collection_name=self.chroma_collection,
                embedding_function=self.embedding,
                persist_directory=self.chroma_persist_directory,
            )
            _ = store.similarity_search("ping", k=1)

        elif self.store_type == "opensearch":
            from langchain_community.vectorstores import OpenSearchVectorSearch

            store = OpenSearchVectorSearch(
                opensearch_url=self.opensearch_url,
                index_name=self.opensearch_index_name,
                embedding_function=self.embedding,
                engine="faiss",
            )
            _ = store.similarity_search("ping", k=1)

        elif self.store_type == "cassandra":
            from langchain_community.vectorstores.cassandra import Cassandra
            from cassandra.cluster import Cluster

            contact_points = [cp.strip() for cp in self.cassandra_contact_points.split(",") if cp.strip()]
            cluster = Cluster(contact_points)
            session = cluster.connect()

            store = Cassandra(
                embedding=self.embedding,
                session=session,
                keyspace=self.cassandra_keyspace,
                table_name=self.cassandra_table_name,
            )
            _ = store.similarity_search("ping", k=1)
            session.shutdown()

        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")

    @asynccontextmanager
    async def get_store(self) -> AsyncIterator[VectorStore]:
        if self.store_type in ("astra", "astradb"):
            from astrapy.authentication import StaticTokenProvider
            from langchain_astradb import AstraDBVectorStore

            store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.astra_collection,
                namespace=self.astra_keyspace,
                token=StaticTokenProvider(self.astra_token.get_secret_value()),
                api_endpoint=self.astra_endpoint,
            )
            try:
                yield store
            finally:
                logger.debug("Cleaning up Astra store")

        elif self.store_type == "chroma":
            from langchain_chroma import Chroma

            store = Chroma(
                collection_name=self.chroma_collection,
                embedding_function=self.embedding,
                persist_directory=self.chroma_persist_directory,
            )
            yield store

        elif self.store_type == "opensearch":
            from langchain_community.vectorstores import OpenSearchVectorSearch

            store = OpenSearchVectorSearch(
                opensearch_url=self.opensearch_url,
                index_name=self.opensearch_index_name,
                embedding_function=self.embedding,
                engine="faiss",
            )
            yield store

        elif self.store_type == "cassandra":
            from langchain_community.vectorstores.cassandra import Cassandra
            from cassandra.cluster import Cluster

            contact_points = [cp.strip() for cp in self.cassandra_contact_points.split(",") if cp.strip()]
            cluster = Cluster(contact_points)
            session = cluster.connect()

            store = Cassandra(
                embedding=self.embedding,
                session=session,
                keyspace=self.cassandra_keyspace,
                table_name=self.cassandra_table_name,
            )
            try:
                yield store
            finally:
                session.shutdown()
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
