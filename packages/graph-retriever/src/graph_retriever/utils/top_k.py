import heapq
from collections.abc import Iterable
from typing import cast

from graph_retriever.content import Content
from graph_retriever.utils.math import cosine_similarity_top_k


def top_k(
    batches: Iterable[list[Content]],
    *,
    embedding: list[float],
    k: int,
    are_batches_sorted: bool = False,
) -> list[Content]:
    """
    Select the top-k contents from the given batches.

    If all contents have scores, they will be used for the comparison rather
    than being computed.

    Parameters
    ----------
    batches : Iterable[list[Content]]
        The batches of content to select the top-K from.
    embedding: list[float]
        The embedding we're looking for.
    k : int
        The number of items to select.
    are_batches_sorted : bool, default False
        Whether the content of each batch are sorted. If true, and all content
        has scores, a more efficient top-K selection will be used which doesn't
        need to consider all of each batch.

    Returns
    -------
    list[Content]
        Top-K by similarity. All results will have their `score` set.
    """
    # TODO: Consider passing threshold here to limit results.

    if all(c.score is not None for batch in batches for c in batch):
        sorted_items: Iterable[Content]
        if are_batches_sorted:
            sorted_items = heapq.merge(*batches, key=_score, reverse=True)
        else:
            sorted_items = sorted(
                [c for batch in batches for c in batch], key=_score, reverse=True
            )

        # Use a dict as a simple way to de-duplicate by ID.
        # We may be able to rely on all values with the same score
        # appearing adjacently (and avoid the dict/set), but we'd
        # also need to ensure that if two different IDs have the same
        # score, we don't have `A, B, A`, etc.
        results: dict[str, Content] = {}
        for c in sorted_items:
            results.setdefault(c.id, c)
            if len(results) >= k:
                break

        return list(results.values())
    else:
        return _similarity_sort_top_k(batches, embedding=embedding, k=k)


def _score(content: Content) -> float:
    return cast(float, content.score)


def _similarity_sort_top_k(
    batches: Iterable[list[Content]], *, embedding: list[float], k: int
) -> list[Content]:
    # Flatten the content and use a dict to deduplicate.
    # We need to do this *before* selecting the top_k to ensure we don't
    # get duplicates (and fail to produce `k`).
    flat = list({c.id: c for batch in batches for c in batch}.values())

    top_k, scores = cosine_similarity_top_k(
        [embedding], [c.embedding for c in flat], top_k=k
    )

    results = []
    for (_x, y), score in zip(top_k, scores):
        content = flat[y]
        content.score = score
        results.append(content)
    return results
