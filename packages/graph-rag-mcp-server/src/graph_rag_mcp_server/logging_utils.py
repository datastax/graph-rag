import logging
import os
import sys


def configure_debug_logging(logger_name: str = "graph_rag_mcp_server") -> logging.Logger:
    # 1. Set root logger to DEBUG (to capture all loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("astrapy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 2. Determine output stream based on environment
    is_mcp_stdio = not sys.stdout.isatty()
    stream_target = sys.stderr if is_mcp_stdio else sys.stdout

    # 3. Stream handler for console/MCP inspector
    stream_handler = logging.StreamHandler(stream_target)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 4. Optional: file handler for full debug logs
    if os.getenv("DEBUG_TO_FILE", "false").lower() in ("1", "true", "yes"):
        file_handler = logging.FileHandler("debug.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 5. Return module-specific logger (optional)
    return logging.getLogger(logger_name)
