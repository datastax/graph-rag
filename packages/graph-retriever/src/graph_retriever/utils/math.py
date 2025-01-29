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
    X : Matrix
        A matrix of shape (m, n), where `m` is the number of rows and `n` is the
        number of columns (features).
    Y : Matrix
        A matrix of shape (p, n), where `p` is the number of rows and `n` is the
        number of columns (features).

    Returns
    -------
    np.ndarray
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
