from dataclasses import dataclass
from typing import Generic, Iterator, Tuple, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Offset:
    """Class for tracking a position in the iteraiton."""
    index: int

class PersistentIteration(Generic[T]):
    def __init__(self,
                 journal_name: str,
                 iterator: Iterator[T]) -> None:
        """Create a persistent iteration.

        This creates a journal file with the name `journal_name` containing the indices
        of completed items. When resuming iteration, the already processed indices will
        be skipped.

        Args:
            journal_name: Name of the journal file to use. If it doesn't exist it will
                be created. The indices of completed items will be written to the journal.
            iterator: The iterator to process persistently. It must be deterministic --
                elements should always be returned in the same order on restarts.
        """
        self.iterator = enumerate(iterator)
        self.pending = {}

        self._completed = set()
        try:
            read_journal = open(journal_name)
            for line in read_journal:
                self._completed.add(Offset(index = int(line)))
        except FileNotFoundError:
            pass

        self._write_journal = open(journal_name, "a")

    def __next__(self) -> Tuple[Offset, T]:
        index, item = next(self.iterator)
        offset = Offset(index)

        while offset in self._completed:
            index, item = next(self.iterator)
            offset = Offset(index)

        self.pending[offset] = item
        return (offset, item)

    def __iter__(self) -> Iterator[Tuple[Offset, T]]:
        return self

    def ack(self, offset: Offset) -> int:
        self._write_journal.write(f"{offset.index}\n")
        self._write_journal.flush()
        self._completed.add(offset)

        self.pending.pop(offset)
        return len(self.pending)

    def pending_count(self) -> int:
        return len(self.pending)

    def completed_count(self) -> int:
        return len(self._completed)