from collections.abc import Iterable
from typing import cast

from graph_retriever.content import Content
from graph_retriever.utils.math import cosine_similarity_top_k


def top_k(
    contents: Iterable[Content],
    *,
    embedding: list[float],
    k: int,
) -> list[Content]:
    """
    Select the top-k contents from the given contet.

    Parameters
    ----------
    contents :
        The content from which to select the top-K.
    embedding: list[float]
        The embedding we're looking for.
    k :
        The number of items to select.

    Returns
    -------
    list[Content]
        Top-K by similarity. All results will have their `score` set.
    """
    # TODO: Consider handling specially cases of already-sorted batches (merge).
    # TODO: Consider passing threshold here to limit results.

    # Use dicts to de-duplicate by ID. This ensures we choose the top K distinct
    # content (rather than K copies of the same content).
    scored = {c.id: c for c in contents if c.score is not None}
    unscored = {c.id: c for c in contents if c.score is None if c.id not in scored}

    if unscored:
        top_unscored = _similarity_sort_top_k(
            list(unscored.values()), embedding=embedding, k=k
        )
        scored.update(top_unscored)

    sorted = list(scored.values())
    sorted.sort(key=_score, reverse=True)

    return sorted[:k]


def _score(content: Content) -> float:
    return cast(float, content.score)


def _similarity_sort_top_k(
    contents: list[Content], *, embedding: list[float], k: int
) -> dict[str, Content]:
    # Flatten the content and use a dict to deduplicate.
    # We need to do this *before* selecting the top_k to ensure we don't
    # get duplicates (and fail to produce `k`).
    top_k, scores = cosine_similarity_top_k(
        [embedding], [c.embedding for c in contents], top_k=k
    )

    results = {}
    for (_x, y), score in zip(top_k, scores):
        c = contents[y]
        c.score = score
        results[c.id] = c
    return results
