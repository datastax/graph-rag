# start with: DEBUG_TO_FILE=true uv run mcp dev -e . src/graph_rag_mcp_server/server.py

from typing import Any
from pydantic import ValidationError
from graph_rag_mcp_server.logging_utils import configure_debug_logging
logger = configure_debug_logging(__name__)

from importlib.resources import files

from mcp.server.fastmcp import Context, FastMCP

from graph_rag_mcp_server.logging_utils import configure_debug_logging

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from graph_rag_mcp_server.config import ServerConfig

from langchain_graph_retriever.graph_retriever import GraphRetriever

try:
    config = ServerConfig()
    config.validate_connections()
    logger.info("Server config loaded and validated successfully")
except ValidationError as e:
    logger.error("Config validation failed.")
    for err in e.errors():
        logger.error(f"Field: {err['loc']}, Error: {err['msg']}")
    raise SystemExit(1)
except Exception:
    logger.exception("Failed to load or validate server config")
    raise

@dataclass
class AppContext:
    store: VectorStore


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    logger.debug("Initializing application context")

    try:
        async with config.get_store() as store:
            yield AppContext(store=store)
    except Exception as e:
        logger.exception("Error during app_lifespan startup")
        raise

# Specify dependencies for deployment and development
mcp = FastMCP("Graph RAG", dependencies=["langchain-graph-retriever"], lifespan=app_lifespan)

logger.debug("MCP server loaded")


def load_resource(file_name: str) -> str:
    path = files("graph_rag_mcp_server.resources").joinpath(file_name)
    logger.debug(f"Loading resource at: {path}")
    return path.read_text(encoding="utf-8")

@mcp.resource("config://test-log")
def test_log_resource() -> str:
    logger.debug("Inside test_log_resource")
    return "Logging worked!"

@mcp.resource("config://getting_started", name="Getting Started", description="An intro to the GraphRAG MCP Server and its capabilities", mime_type="text/markdown")
def get_getting_started() -> str:
    logger.debug("get_getting_started called")
    return load_resource("getting_started.md")

@mcp.resource("config://strategy_options/{strategy}", name="Strategy Options", description="The list of options available when using a specific strategy (`eager` or `mmr`)", mime_type="application/json")
def get_strategy_options(strategy: str) -> str:
    logger.debug(f"get_strategy_options called with: {strategy}")
    if strategy not in ["eager", "mmr"]:
        raise ValueError(f"Invalid strategy: {strategy}")
    return load_resource(f"config_{strategy}.json")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
async def retrieve(ctx: Context, query: str, options: dict[str, Any] = {}) -> list[Document]:
    """Retrieve documents from the store"""
    logger.debug(f"Retrieving  documents for query: {query}")
    if not query:
        raise ValueError("Query cannot be empty")

    store:VectorStore = ctx.request_context.lifespan_context.store

    retriever = GraphRetriever(store=store, edges=options.get("edges", []))

    return await retriever.ainvoke(query, **options)
