from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass
class Content:
    """
    Model representing retrieved content.

    Parameters
    ----------
    id :
        The ID of the content.
    content :
        The content.
    embedding :
        The embedding of the content.
    metadata :
        The metadata associated with the content.
    mime_type :
        The MIME type of the content.
    """

    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    mime_type: str = "text/plain"

    @staticmethod
    def new(
        id: str,
        content: str,
        embedding: list[float] | Callable[[str], list[float]],
        *,
        metadata: dict[str, Any] | None = None,
        mime_type: str = "text/plain",
    ) -> Content:
        """
        Create a new content.

        Parameters
        ----------
        id :
            The ID of the content.
        content :
            The content.
        embedding :
            The embedding, or a function to apply to the content to compute the
            embedding.
        metadata :
            The metadata associated with the content.
        mime_type :
            The MIME type of the content.

        Returns
        -------
        :
            The created content.
        """
        return Content(
            id=id,
            content=content,
            embedding=embedding(content) if callable(embedding) else embedding,
            metadata=metadata or {},
            mime_type=mime_type,
        )
