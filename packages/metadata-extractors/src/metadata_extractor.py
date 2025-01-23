from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Set, TypeVar

InputT = TypeVar("InputT")

class AbstractMetadataExtractor(ABC, Generic[InputT]):
    """Interface for extracting links (incoming, outgoing, bidirectional)."""

    @abstractmethod
    def load_default_model(embedding_model: str) -> Any:
        """Load the default model for the extractor.

        Args:
            embedding_model: The name of the embedding model to load.

        Returns:
            The default model for the extractor.
        """

    @abstractmethod
    def extract_one(self, input: InputT) -> Set[Any]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            input: The input content to extract edges from.

        Returns:
            Set of links extracted from the input.
        """
        
    @abstractmethod
    def extract_many(self, inputs: Iterable[InputT]) -> Iterable[Set[Any]]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            inputs: The input content to extract edges from.

        Returns:
            Iterable over the set of links extracted from the input.
        """
        return map(self.extract_one, inputs)

