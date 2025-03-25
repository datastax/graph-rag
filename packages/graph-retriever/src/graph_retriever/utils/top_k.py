from collections.abc import Iterable

from graph_retriever.content import Content
from graph_retriever.utils.math import cosine_similarity_top_k


def top_k(
    contents: Iterable[Content],
    *,
    embedding: list[float],
    k: int,
) -> list[Content]:
    """
    Select the top-k contents from the given content.

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
        Top-K by similarity.
    """
    # TODO: Consider handling specially cases of already-sorted batches (merge).
    # TODO: Consider passing threshold here to limit results.

    # Use dicts to de-duplicate by ID. This ensures we choose the top K distinct
    # content (rather than K copies of the same content).
    unscored = {c.id: c for c in contents}

    top_scored = _similarity_sort_top_k(
        list(unscored.values()), embedding=embedding, k=k
    )

    sorted = list(top_scored.values())
    sorted.sort(key=_score, reverse=True)

    return [c[0] for c in sorted]


def _score(content_with_score: tuple[Content, float]) -> float:
    return content_with_score[1]


def _similarity_sort_top_k(
    contents: list[Content], *, embedding: list[float], k: int
) -> dict[str, tuple[Content, float]]:
    # Flatten the content and use a dict to deduplicate.
    # We need to do this *before* selecting the top_k to ensure we don't
    # get duplicates (and fail to produce `k`).
    top_k, scores = cosine_similarity_top_k(
        [embedding], [c.embedding for c in contents], top_k=k
    )

    results = {}
    for (_x, y), score in zip(top_k, scores):
        c = contents[y]
        results[c.id] = (c, score)
    return results
