from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class Content(BaseModel):
    """
    Model representing retrieved content.

    Attributes
    ----------
    id : str
        The ID of the content.
    content : str
        The content.
    embedding : list[float]
        The embedding of the content.
    metadata : dict[str, Any]
        The metadata associated with the content.
    mime_type : str
        The MIME type of the content.
    """

    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = {}

    mime_type: str = "text/plain"

    @staticmethod
    def new(
        id: str,
        content: str,
        embedding: list[float] | Callable[[str], list[float]],
        metadata: dict[str, Any] = {},
        mime_type: str = "text/plain",
    ) -> Content:
        """
        Create a new content.

        Parameters
        ----------
        id : str
            The ID of the content.
        content : str
            The content.
        embedding : list[float] | Callable[[str], list[float]]
            The embedding, or a function to apply to the content to compute the
            embedding.
        metadata : dict[str, Any], optional
            The metadata associated with the content.
        mime_type : str, optional
            The MIME type of the content.

        Returns
        -------
        Content
            The created content.
        """
        return Content(
            id=id,
            content=content,
            embedding=embedding(content) if callable(embedding) else embedding,
            metadata=metadata,
            mime_type=mime_type,
        )
