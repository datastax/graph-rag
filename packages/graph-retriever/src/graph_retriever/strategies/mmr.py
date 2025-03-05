"""Provide MMR (max-marginal-relevance) traversal strategy."""

import dataclasses
from collections.abc import Iterable
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from graph_retriever.strategies.base import NodeTracker, Strategy
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


@dataclasses.dataclass
class Mmr(Strategy):
    """
    Maximal Marginal Relevance (MMR) traversal strategy.

    This strategy selects nodes by balancing relevance to the query and diversity
    among the results. It uses a `lambda_mult` parameter to control the trade-off
    between relevance and redundancy. Nodes are scored based on their similarity
    to the query and their distance from already selected nodes.

    Parameters
    ----------
    select_k :
        Maximum number of nodes to retrieve during traversal.
    start_k :
        Number of documents to fetch via similarity for starting the traversal.
        Added to any initial roots provided to the traversal.
    adjacent_k :
        Number of documents to fetch for each outgoing edge.
    max_depth :
        Maximum traversal depth. If `None`, there is no limit.
    lambda_mult :
        Controls the trade-off between relevance and diversity. A value closer
        to 1 prioritizes relevance, while a value closer to 0 prioritizes
        diversity. Must be between 0 and 1 (inclusive).
    min_mmr_score :
        Only nodes with a score greater than or equal to this value will be
        selected.
    k:
        Deprecated: Use `select_k` instead.
        Maximum number of nodes to select and return during traversal.
    """

    lambda_mult: float = 0.5
    min_mmr_score: float = NEG_INF

    _selected_ids: list[str] = dataclasses.field(default_factory=list)
    """List of selected IDs (in selection order)."""

    _candidate_id_to_index: dict[str, int] = dataclasses.field(default_factory=dict)
    """Dictionary of candidate IDs to indices in candidates and candidate_embeddings."""

    _candidates: list[_MmrCandidate] = dataclasses.field(default_factory=list)
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
        return np.ndarray((self.select_k, self._dimensions), dtype=np.float32)

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
        candidate_id :
            The ID of the candidate to pop.

        Returns
        -------
        candidate :
            The candidate with the given ID.
        embedding :
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

    def _next(self) -> Node | None:
        """
        Select and pop the best item being considered.

        Updates the consideration set based on it.

        Returns
        -------
            The best node available or None if none are available.
        """
        if self._best_id is None or self._best_score < self.min_mmr_score:
            return None

        # Get the selection and remove from candidates.
        selected_id = self._best_id
        selected, selected_embedding = self._pop_candidate(selected_id)

        # Add the ID and embedding to the selected information.
        selection_index = len(self._selected_ids)
        self._selected_ids.append(selected_id)
        self._selected_embeddings[selection_index] = selected_embedding

        # Create the selected result node.
        selected_node = selected.node
        selected_node.extra_metadata["_mmr_score"] = selected.score
        selected_node.extra_metadata["_redundancy"] = (
            selected.weighted_redundancy / self._lambda_mult_complement
        )

        # Reset the best score / best ID.
        self._best_score = NEG_INF
        self._best_id = None

        # Update the candidates redundancy, tracking the best node.
        if self._candidate_embeddings.shape[0] > 0:
            similarity = cosine_similarity(
                self._candidate_embeddings, np.expand_dims(selected_embedding, axis=0)
            )
            for index, candidate in enumerate(self._candidates):
                candidate.update_redundancy(
                    self._lambda_mult_complement * similarity[index][0]
                )
                if candidate.score > self._best_score:
                    self._best_score = candidate.score
                    self._best_id = candidate.node.id

        return selected_node

    @override
    def iteration(self, nodes: Iterable[Node], tracker: NodeTracker) -> None:
        """Add candidates to the consideration set."""
        node_count = len(list(nodes))
        if node_count > 0:
            # Build up a matrix of the remaining candidate embeddings.
            # And add them to the candidate set
            new_embeddings: NDArray[np.float32] = np.ndarray(
                (
                    node_count,
                    self._dimensions,
                )
            )
            offset = self._candidate_embeddings.shape[0]
            for index, candidate_node in enumerate(nodes):
                self._candidate_id_to_index[candidate_node.id] = offset + index
                new_embeddings[index] = candidate_node.embedding

            # Compute the similarity to the query.
            similarity = cosine_similarity(new_embeddings, self._nd_query_embedding)

            # Compute the distance metrics of all of pairs in the selected set with
            # the new candidates.
            redundancy = cosine_similarity(
                new_embeddings, self._already_selected_embeddings()
            )
            for index, candidate_node in enumerate(nodes):
                max_redundancy = 0.0
                if redundancy.shape[0] > 0:
                    max_redundancy = redundancy[index].max()
                candidate = _MmrCandidate(
                    node=candidate_node,
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

        while tracker.num_remaining > 0:
            next = self._next()
            if next is None:
                break

            num_traversing = tracker.select_and_traverse([next])
            if num_traversing == 1:
                break
