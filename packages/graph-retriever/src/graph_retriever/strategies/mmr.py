"""Provide MMR (max-marginal-relevance) traversal strategy."""

import dataclasses
from collections.abc import Iterable
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from typing_extensions import override

from graph_retriever.strategies.base import Strategy
from graph_retriever.types import Node
from graph_retriever.utils.math import cosine_similarity

NEG_INF = float("-inf")


def _emb_to_ndarray(embedding: list[float]) -> NDArray[np.float32]:
    emb_array = np.array(embedding, dtype=np.float32)
    if emb_array.ndim == 1:
        emb_array = np.expand_dims(emb_array, axis=0)
    return emb_array


@dataclasses.dataclass
class _MmrCandidate:
    node: Node
    similarity: float
    weighted_similarity: float
    weighted_redundancy: float
    score: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.score = self.weighted_similarity - self.weighted_redundancy

    def update_redundancy(self, new_weighted_redundancy: float) -> None:
        if new_weighted_redundancy > self.weighted_redundancy:
            self.weighted_redundancy = new_weighted_redundancy
            self.score = self.weighted_similarity - self.weighted_redundancy


class Mmr(Strategy):
    """
    Maximal Marginal Relevance (MMR) traversal strategy.

    This strategy selects nodes by balancing relevance to the query and diversity
    among the results. It uses a `lambda_mult` parameter to control the trade-off
    between relevance and redundancy. Nodes are scored based on their similarity
    to the query and their distance from already selected nodes.

    Parameters
    ----------
    k : int, default 5
        Maximum number of nodes to retrieve during traversal.
    start_k : int, default 4
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k : int, default 10
        Number of documents to fetch for each outgoing edge.
    max_depth : int, optional
        Maximum traversal depth. If `None`, there is no limit.
    lambda_mult : float, default 0.5
        Controls the trade-off between relevance and diversity. A value closer
        to 1 prioritizes relevance, while a value closer to 0 prioritizes
        diversity. Must be between 0 and 1 (inclusive).
    score_threshold : float, default -infinity
        Only nodes with a score greater than or equal to this value will be
        selected.

    Attributes
    ----------
    k : int
        Maximum number of nodes to retrieve during traversal.
    start_k : int
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k : int
        Number of documents to fetch for each outgoing edge.
    max_depth : int
        Maximum traversal depth. If `None`, there is no limit.
    lambda_mult : float
        Controls the trade-off between relevance and diversity. A value closer
        to 1 prioritizes relevance, while a value closer to 0 prioritizes
        diversity. Must be between 0 and 1 (inclusive).
    score_threshold : float
        Only nodes with a score greater than or equal to this value will be
        selected.
    """

    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float = NEG_INF

    _selected_ids: list[str] = []
    """List of selected IDs (in selection order)."""

    _candidate_id_to_index: dict[str, int] = {}
    """Dictionary of candidate IDs to indices in candidates and candidate_embeddings."""

    _candidates: list[_MmrCandidate] = []
    """List containing information about candidates.
    Same order as rows in `candidate_embeddings`.
    """

    _best_score: float = NEG_INF
    _best_id: str | None = None

    @cached_property
    def _nd_query_embedding(self) -> NDArray[np.float32]:
        assert self._query_embedding, (
            "shouldn't access embedding / dimensions until initialized"
        )
        return _emb_to_ndarray(self._query_embedding)

    @property
    def _dimensions(self) -> int:
        return self._nd_query_embedding.shape[1]

    @property
    def _lambda_mult_complement(self) -> float:
        return 1 - self.lambda_mult

    @cached_property
    def _selected_embeddings(self) -> NDArray[np.float32]:
        """
        (N, dim) ndarray with a row for each selected node.

        Returns
        -------
        NDArray[np.float32]
            (N, dim) ndarray with a row for each selected node.
        """
        return np.ndarray((self.k, self._dimensions), dtype=np.float32)

    @cached_property
    def _candidate_embeddings(self) -> NDArray[np.float32]:
        """
        (N, dim) ndarray with a row for each candidate.

        Returns
        -------
        NDArray[np.float32]
            (N, dim) ndarray with a row for each candidate.
        """
        return np.ndarray((0, self._dimensions), dtype=np.float32)

    def candidate_ids(self) -> Iterable[str]:
        """
        Return the IDs of the candidates.

        Returns
        -------
        Iterable[str]
            The IDs of the candidates.
        """
        return self._candidate_id_to_index.keys()

    def _already_selected_embeddings(self) -> NDArray[np.float32]:
        """
        Return the selected embeddings sliced to the already assigned values.

        Returns
        -------
        NDArray[np.float32]
            The selected embeddings sliced to the already assigned values.
        """
        selected = len(self._selected_ids)
        return np.vsplit(self._selected_embeddings, [selected])[0]

    def _pop_candidate(
        self, candidate_id: str
    ) -> tuple[_MmrCandidate, NDArray[np.float32]]:
        """
        Pop the candidate with the given ID.

        Parameters
        ----------
        candidate_id : str
            The ID of the candidate to pop.

        Returns
        -------
        candidate : _MmrCandidate
            The candidate with the given ID.
        embedding : NDArray[np.float32]
            The `NDArray` embedding of the candidate.

        Raises
        ------
        ValueError
            If `self._candidates[self._candidate_id_to_index[candidate_id]].id`
            does not match `candidate_id`. This would indicate an internal
            book-keeping error in the positions of candidates.
        """
        # Get the embedding for the id.
        index = self._candidate_id_to_index.pop(candidate_id)
        candidate = self._candidates[index]
        if candidate.node.id != candidate_id:
            msg = (
                "ID in self.candidate_id_to_index doesn't match the ID of the "
                "corresponding index in self.candidates"
            )
            raise ValueError(msg)
        embedding: NDArray[np.float32] = self._candidate_embeddings[index].copy()

        # Swap that index with the last index in the candidates and
        # candidate_embeddings.
        last_index = self._candidate_embeddings.shape[0] - 1

        if index == last_index:
            self._candidates.pop()
        else:
            self._candidate_embeddings[index] = self._candidate_embeddings[last_index]

            old_last = self._candidates.pop()
            self._candidates[index] = old_last
            self._candidate_id_to_index[old_last.node.id] = index

        self._candidate_embeddings = np.vsplit(
            self._candidate_embeddings, [last_index]
        )[0]

        return candidate, embedding

    @override
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """
        Select and pop the best item being considered.

        Updates the consideration set based on it.

        Returns
        -------
            A tuple containing the ID of the best item.
        """
        if limit == 0:
            return []
        if self._best_id is None or self._best_score < self.score_threshold:
            return []

        # Get the selection and remove from candidates.
        selected_id = self._best_id
        selected, selected_embedding = self._pop_candidate(selected_id)

        # Add the ID and embedding to the selected information.
        selection_index = len(self._selected_ids)
        self._selected_ids.append(selected_id)
        self._selected_embeddings[selection_index] = selected_embedding

        # Create the selected result node.
        selected_node = selected.node
        selected_node.extra_metadata = {
            "_similarity_score": selected.similarity,
            "_mmr_score": self._best_score,
        }

        # Reset the best score / best ID.
        self._best_score = NEG_INF
        self._best_id = None

        # Update the candidates redundancy, tracking the best node.
        if self._candidate_embeddings.shape[0] > 0:
            similarity = cosine_similarity(
                self._candidate_embeddings, np.expand_dims(selected_embedding, axis=0)
            )
            for index, candidate in enumerate(self._candidates):
                candidate.update_redundancy(similarity[index][0])
                if candidate.score > self._best_score:
                    self._best_score = candidate.score
                    self._best_id = candidate.node.id

        return [selected_node]

    @override
    def discover_nodes(self, nodes: dict[str, Node]) -> None:
        """Add candidates to the consideration set."""
        # Determine the keys to actually include.
        # These are the candidates that aren't already selected
        # or under consideration.

        include_ids_set = set(nodes.keys())
        include_ids_set.difference_update(self._selected_ids)
        include_ids_set.difference_update(self._candidate_id_to_index.keys())
        include_ids = list(include_ids_set)

        # Now, build up a matrix of the remaining candidate embeddings.
        # And add them to the
        new_embeddings: NDArray[np.float32] = np.ndarray(
            (
                len(include_ids),
                self._dimensions,
            )
        )
        offset = self._candidate_embeddings.shape[0]
        for index, candidate_id in enumerate(include_ids):
            self._candidate_id_to_index[candidate_id] = offset + index
            new_embeddings[index] = nodes[candidate_id].embedding

        # Compute the similarity to the query.
        similarity = cosine_similarity(new_embeddings, self._nd_query_embedding)

        # Compute the distance metrics of all of pairs in the selected set with
        # the new candidates.
        redundancy = cosine_similarity(
            new_embeddings, self._already_selected_embeddings()
        )
        for index, candidate_id in enumerate(include_ids):
            max_redundancy = 0.0
            if redundancy.shape[0] > 0:
                max_redundancy = redundancy[index].max()
            candidate = _MmrCandidate(
                node=nodes[candidate_id],
                similarity=similarity[index][0],
                weighted_similarity=self.lambda_mult * similarity[index][0],
                weighted_redundancy=self._lambda_mult_complement * max_redundancy,
            )
            self._candidates.append(candidate)

            if candidate.score >= self._best_score:
                self._best_score = candidate.score
                self._best_id = candidate.node.id

        # Add the new embeddings to the candidate set.
        self._candidate_embeddings = np.vstack(
            (
                self._candidate_embeddings,
                new_embeddings,
            )
        )
