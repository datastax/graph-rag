import asyncio
import zipfile
from collections.abc import Callable, Iterable, Iterator
from math import ceil
from os.path import dirname
from os.path import join as joinpath

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

PARA_WITH_HYPERLINK = joinpath(dirname(__file__), "para_with_hyperlink.zip")


def wikipedia_lines() -> Iterable[str]:
    """
    Return iterable of lines from the wikipedia file.

    Yields
    ------
    str
        Lines from the Wikipedia file.
    """
    with zipfile.ZipFile(PARA_WITH_HYPERLINK, "r") as archive:
        with archive.open("para_with_hyperlink.jsonl", "r") as para_with_hyperlink:
            for line in para_with_hyperlink:
                yield str(line)


BATCH_SIZE = 1000
MAX_IN_FLIGHT = 5

EXCEPTIONS_TO_RETRY = (
    httpx.NetworkError,
    astrapy.exceptions.DataAPIException,
)

MAX_RETRIES = 8

BatchPreparer = Callable[[Iterator[str]], Iterator[Document]]


async def aload_2wikimultihop(store: VectorStore, batch_prepare: BatchPreparer) -> None:
    """
    Load 2wikimultihop data into the given `VectorStore`.

    Parameters
    ----------
    store : VectorStore
        The VectorStore to populate.
    batch_prepare : BatchPreparer
        Function to apply to batches of lines to produce the document.
    """
    persistence = PersistentIteration(
        journal_name="load_2wikimultihop.jrnl",
        iterator=batched(wikipedia_lines(), BATCH_SIZE),
    )
    total_batches = ceil(LINES_IN_FILE / BATCH_SIZE) - persistence.completed_count()
    if persistence.completed_count() > 0:
        print(  # noqa: T201
            f"Resuming loading with {persistence.completed_count()}"
            f" completed, {total_batches} remaining"
        )
    async with asyncio.TaskGroup() as tg:
        tasks = []

        @backoff.on_exception(
            backoff.expo,
            EXCEPTIONS_TO_RETRY,
            max_tries=MAX_RETRIES,
        )
        async def add_docs(batch_docs, offset) -> None:
            await store.aadd_documents(batch_docs)
            persistence.ack(offset)

        for offset, batch_lines in tqdm(persistence, total=total_batches):
            batch_docs = batch_prepare(batch_lines)
            if batch_docs:
                tasks.append(tg.create_task(add_docs(batch_docs, offset)))
                while len(tasks) >= MAX_IN_FLIGHT:
                    _, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    tasks = list(pending)
            else:
                persistence.ack(offset)

    assert persistence.pending_count() == 0
