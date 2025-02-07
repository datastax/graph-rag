"""
Package containing useful Document Transformers.

Many of these add metadata that could be useful for linking content, such as
extracting named entities or keywords from the page content.

Also includes a transformer for shredding metadata, for use with stores
that do not support querying on elements of lists.
"""

from .parent import ParentTransformer
from .shredding import ShreddingTransformer

__all__ = [
    "ShreddingTransformer",
    "ParentTransformer",
]
