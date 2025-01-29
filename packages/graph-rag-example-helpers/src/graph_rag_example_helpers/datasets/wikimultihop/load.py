import asyncio
import os.path
import zipfile
from collections.abc import Callable, Iterable, Iterator
from math import ceil

import astrapy
import astrapy.exceptions
import backoff
import httpx
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm  # type: ignore[import-untyped]

from graph_rag_example_helpers.batched import batched
from graph_rag_example_helpers.persistent_iteration import PersistentIteration

LINES_IN_FILE = 5989847


def wikipedia_lines(para_with_hyperlink_zip_path: str) -> Iterable[bytes]:
    """
    Return iterable of lines from the wikipedia file.

    Parameters
    ----------
    para_with_hyperlink_zip_path : str
        Path to `para_with_hyperlink.zip` downloaded following the instructions
        in
        [2wikimultihop](https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file#new-update-april-7-2021).

    Yields
    ------
    str
        Lines from the Wikipedia file.
    """
    with zipfile.ZipFile(para_with_hyperlink_zip_path, "r") as archive:
        with archive.open("para_with_hyperlink.jsonl", "r") as para_with_hyperlink:
            yield from para_with_hyperlink


BATCH_SIZE = 512
MAX_IN_FLIGHT = 1

EXCEPTIONS_TO_RETRY = (
    httpx.TransportError,
    astrapy.exceptions.DataAPIException,
)

MAX_RETRIES = 8

BatchPreparer = Callable[[Iterator[bytes]], Iterator[Document]]


async def aload_2wikimultihop(
    para_with_hyperlink_zip_path: str, store: VectorStore, batch_prepare: BatchPreparer
) -> None:
    """
    Load 2wikimultihop data into the given `VectorStore`.

    Parameters
    ----------
    para_with_hyperlink_zip_path : str
        Path to `para_with_hyperlink.zip` downloaded following the instructions
        in
        [2wikimultihop](https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file#new-update-april-7-2021).
    store : VectorStore
        The VectorStore to populate.
    batch_prepare : BatchPreparer
        Function to apply to batches of lines to produce the document.
    """
    assert os.path.isfile(para_with_hyperlink_zip_path)
    persistence = PersistentIteration(
        journal_name="load_2wikimultihop.jrnl",
        iterator=batched(wikipedia_lines(para_with_hyperlink_zip_path), BATCH_SIZE),
    )
    total_batches = ceil(LINES_IN_FILE / BATCH_SIZE) - persistence.completed_count()
    if persistence.completed_count() > 0:
        print(  # noqa: T201
            f"Resuming loading with {persistence.completed_count()}"
            f" completed, {total_batches} remaining"
        )

    # We can't use asyncio.TaskGroup in 3.10. This would be simpler with that.
    tasks: list[asyncio.Task] = []

    @backoff.on_exception(
        backoff.expo,
        EXCEPTIONS_TO_RETRY,
        max_tries=MAX_RETRIES,
    )
    async def add_docs(batch_docs, offset) -> None:
        from astrapy.exceptions import InsertManyException

        try:
            await store.aadd_documents(batch_docs)
            persistence.ack(offset)
        except InsertManyException as err:
            for err_desc in err.error_descriptors:
                if err_desc.error_code != "DOCUMENT_ALREADY_EXISTS":
                    print(err_desc)  # noqa: T201
            raise

    for offset, batch_lines in tqdm(persistence, total=total_batches):
        batch_docs = batch_prepare(batch_lines)
        if batch_docs:
            ids = [doc.id for doc in batch_docs]
            assert len(set(ids)) == len(ids)
            task = asyncio.create_task(add_docs(batch_docs, offset))

            # It is OK if tasks are lost upon failure since that means we're
            # aborting the loading.
            tasks.append(task)

            while len(tasks) >= MAX_IN_FLIGHT:
                completed, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for complete in completed:
                    if (e := complete.exception()) is not None:
                        print(f"Exception in task: {e}")  # noqa: T201
                tasks = list(pending)
        else:
            persistence.ack(offset)

    # Make sure all the tasks are done.
    # This wouldn't be necessary if we used a taskgroup, but that is Python 3.11+.
    while len(tasks) > 0:
        completed, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for complete in completed:
            if (e := complete.exception()) is not None:
                print(f"Exception in task: {e}")  # noqa: T201
        tasks = list(pending)

    assert len(tasks) == 0
    assert persistence.pending_count() == 0
