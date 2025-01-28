try:
    # Try importing the function from itertools (Python 3.12+)
    from itertools import batched  # type: ignore[attr-defined]
except ImportError:
    from collections.abc import Iterator
    from itertools import islice
    from typing import TypeVar

    # Fallback implementation for older Python versions

    T = TypeVar("T")

    # This is equivalent to `itertools.batched`, but that is only available in 3.12
    def batched(iterable: Iterator[T], n: int) -> Iterator[Iterator[T]]:  # type: ignore[no-redef]
        """
        Equivalent of itertools.batched for pre 3.12.

        Parameters
        ----------
        iterable : Iterator[T]
            Iterator over elements.
        n : int
            Size of batches.

        Yields
        ------
        Iterator[T]
            Yields iterators over elements in successive batches.

        Raises
        ------
        ValueError
            If `n` is less than 1.
        """
        if n < 1:
            raise ValueError("n must be at least one")
        it = iterable
        while batch := tuple(islice(it, n)):
            yield iter(batch)
