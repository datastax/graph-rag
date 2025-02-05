from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Offset:
    """Class for tracking a position in the iteraiton."""

    index: int


class PersistentIteration(Generic[T]):
    """
    Create a persistent iteration.

    This creates a journal file with the name `journal_name` containing the indices
    of completed items. When resuming iteration, the already processed indices will
    be skipped.

    Parameters
    ----------
    journal_name :
        Name of the journal file to use. If it doesn't exist it will be
        created. The indices of completed items will be written to the
        journal.
    iterator :
        The iterator to process persistently. It must be deterministic --
        elements should always be returned in the same order on restarts.
    """

    def __init__(self, journal_name: str, iterator: Iterator[T]) -> None:
        self.iterator = enumerate(iterator)
        self.pending: dict[Offset, T] = {}

        self._completed = set()
        try:
            read_journal = open(journal_name)
            for line in read_journal:
                self._completed.add(Offset(index=int(line)))
        except FileNotFoundError:
            pass

        self._write_journal = open(journal_name, "a")

    def __next__(self) -> tuple[Offset, T]:
        """
        Return the next offset and item.

        Returns
        -------
        offset :
            The offset of the next item. Should be acknowledge after the item
            is finished processing.
        item :
            The next item.
        """
        index, item = next(self.iterator)
        offset = Offset(index)

        while offset in self._completed:
            index, item = next(self.iterator)
            offset = Offset(index)

        self.pending[offset] = item
        return (offset, item)

    def __iter__(self) -> Iterator[tuple[Offset, T]]:
        """
        Iterate over pairs of offsets and elements.

        Returns
        -------
        :
        """
        return self

    def ack(self, offset: Offset) -> int:
        """
        Acknowledge the given offset.

        This should only be called after the elements in that offset have been
        persisted.

        Parameters
        ----------
        offset :
            The offset to acknowledge.

        Returns
        -------
        :
            The numebr of pending elements.
        """
        self._write_journal.write(f"{offset.index}\n")
        self._write_journal.flush()
        self._completed.add(offset)

        self.pending.pop(offset)
        return len(self.pending)

    def pending_count(self) -> int:
        """
        Return the number of pending (not processed) elements.

        Returns
        -------
        :
            The number of pending elements.
        """
        return len(self.pending)

    def completed_count(self) -> int:
        """
        Return the numebr of completed elements.

        Returns
        -------
        :
            The number of completed elements.
        """
        return len(self._completed)
