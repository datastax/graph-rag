"""Math utility functions for vector operations."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

Matrix = list[list[float]] | list[np.ndarray] | np.ndarray


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """
    Compute row-wise cosine similarity between two equal-width matrices.

    Parameters
    ----------
    X :
        A matrix of shape (m, n), where `m` is the number of rows and `n` is the
        number of columns (features).
    Y :
        A matrix of shape (p, n), where `p` is the number of rows and `n` is the
        number of columns (features).

    Returns
    -------
    :
        A matrix of shape (m, p) containing the cosine similarity scores
        between each row of `X` and each row of `Y`.

    Raises
    ------
    ValueError
        If the number of columns in `X` and `Y` are not equal.

    Notes
    -----
    - If the `simsimd` library is available, it will be used for performance
      optimization. Otherwise, the function falls back to a NumPy implementation.
    - Divide-by-zero and invalid values in similarity calculations are replaced
      with 0.0 in the output.
    """
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def cosine_similarity_top_k(
    X: Matrix,
    Y: Matrix,
    top_k: int | None,
    score_threshold: float | None = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Row-wise cosine similarity with optional top-k and score threshold filtering.

    Parameters
    ----------
    X :
        A matrix of shape (m, n), where `m` is the number of rows and `n` is the
        number of columns (features).
    Y :
        A matrix of shape (p, n), where `p` is the number of rows and `n` is the
        number of columns (features).
    top_k :
        Max number of results to return.
    score_threshold:
        Minimum score to return.

    Returns
    -------
    list[tuple[int, int]]
        Two-tuples of indices `(X_idx, Y_idx)` indicating the respective rows in
        `X` and `Y`.
    list[float]
        The corresponding cosine similarities.
    """
    if len(X) == 0 or len(Y) == 0:
        return [], []
    score_array = cosine_similarity(X, Y)
    score_threshold = score_threshold or -1.0
    score_array[score_array < score_threshold] = 0
    top_k = min(top_k or len(score_array), np.count_nonzero(score_array))
    top_k_idxs = np.argpartition(score_array, -top_k, axis=None)[-top_k:]
    top_k_idxs = top_k_idxs[np.argsort(score_array.ravel()[top_k_idxs])][::-1]
    ret_idxs = np.unravel_index(top_k_idxs, score_array.shape)
    scores = score_array.ravel()[top_k_idxs].tolist()
    return list(zip(*ret_idxs)), scores  # type: ignore
