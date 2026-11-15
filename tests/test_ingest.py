import asyncio
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.db import ChunkStrategy, Document, DocumentStatus
from app.services.extractor import UnreadableFileError, UnsupportedFileTypeError
from app.services.ingest import IngestService
from app.services.vector_store import VectorStore
from tests.fakes import COLLECTION, FakeEmbedder

TEXT = (
    "Refunds are accepted within 30 days of purchase. "
    "Items must be unused and in the original packaging. "
    "Shipping takes three to five business days. "
    "International orders may take longer to arrive. "
    "Support is available on weekdays from nine to five. "
) * 3
DATA = TEXT.encode()


async def all_docs(factory: async_sessionmaker[AsyncSession]) -> list[Document]:
    async with factory() as session:
        return list((await session.execute(select(Document))).scalars())


async def point_count(qdrant: AsyncQdrantClient) -> int:
    return (await qdrant.count(COLLECTION, exact=True)).count


async def ingest(service: IngestService, data: bytes = DATA, name: str = "a.txt"):  # type: ignore[no-untyped-def]
    return await service.ingest(name, data, ChunkStrategy.SENTENCE, 200, 40)


async def test_new_upload_is_stored_and_ready(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    result = await ingest(ingest_service)
    assert result.created
    assert result.document.status == DocumentStatus.READY
    assert result.document.num_chunks > 1
    assert await point_count(qdrant) == result.document.num_chunks
    rows = await all_docs(session_factory)
    assert len(rows) == 1 and rows[0].filename == "a.txt"


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
async def test_both_strategies_work(
    strategy: ChunkStrategy, ingest_service: IngestService, qdrant: AsyncQdrantClient
) -> None:
    result = await ingest_service.ingest("a.txt", DATA, strategy, 200, 40)
    assert result.document.chunk_strategy == strategy
    assert await point_count(qdrant) == result.document.num_chunks > 0


async def test_same_file_twice_is_a_duplicate(
    ingest_service: IngestService, qdrant: AsyncQdrantClient, embedder: FakeEmbedder
) -> None:
    first = await ingest(ingest_service)
    calls_before = embedder.text_calls
    second = await ingest(ingest_service, name="renamed.txt")
    assert not second.created
    assert second.document.id == first.document.id
    assert embedder.text_calls == calls_before  # no new work was done
    assert await point_count(qdrant) == first.document.num_chunks


async def test_other_chunk_settings_do_not_bypass_the_duplicate_check(
    ingest_service: IngestService,
) -> None:
    first = await ingest(ingest_service)
    again = await ingest_service.ingest("a.txt", DATA, ChunkStrategy.FIXED, 300, 50)
    assert not again.created and again.document.id == first.document.id


async def test_three_simultaneous_uploads_store_the_file_once(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: FakeEmbedder,
) -> None:
    embedder.delay = 0.3  # keep the winner busy while the others arrive
    results = await asyncio.gather(*(ingest(ingest_service) for _ in range(3)))
    assert sorted(r.created for r in results) == [False, False, True]
    assert len({r.document.id for r in results}) == 1
    assert len(await all_docs(session_factory)) == 1
    winner = next(r for r in results if r.created)
    assert await point_count(qdrant) == winner.document.num_chunks


async def test_embedding_failure_cleans_up(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: FakeEmbedder,
) -> None:
    embedder.fail = True
    with pytest.raises(RuntimeError):
        await ingest(ingest_service)
    rows = await all_docs(session_factory)
    assert [r.status for r in rows] == [DocumentStatus.FAILED]
    assert await point_count(qdrant) == 0


async def test_a_failed_upload_does_not_block_a_retry(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: FakeEmbedder,
) -> None:
    embedder.fail = True
    with pytest.raises(RuntimeError):
        await ingest(ingest_service)
    embedder.fail = False
    retry = await ingest(ingest_service)
    assert retry.created and retry.document.status == DocumentStatus.READY
    statuses = sorted(r.status.value for r in await all_docs(session_factory))
    assert statuses == ["failed", "ready"]
    assert await point_count(qdrant) == retry.document.num_chunks


async def test_unreadable_file_leaves_nothing_behind(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(UnreadableFileError):
        await ingest(ingest_service, b"   \n ", "empty.txt")
    assert await all_docs(session_factory) == []
    assert await point_count(qdrant) == 0


async def test_unsupported_type_leaves_nothing_behind(
    ingest_service: IngestService,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        await ingest(ingest_service, b"data", "report.docx")
    assert await all_docs(session_factory) == []


async def test_delete_removes_vectors_and_row(
    ingest_service: IngestService,
    qdrant: AsyncQdrantClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    result = await ingest(ingest_service)
    assert await ingest_service.delete(result.document.id) is True
    assert await point_count(qdrant) == 0
    assert await all_docs(session_factory) == []
    assert await ingest_service.delete(result.document.id) is False


async def test_delete_unknown_document(ingest_service: IngestService) -> None:
    assert await ingest_service.delete(uuid4()) is False


async def test_delete_only_touches_that_document(
    ingest_service: IngestService, qdrant: AsyncQdrantClient
) -> None:
    first = await ingest(ingest_service, DATA, "one.txt")
    second = await ingest(ingest_service, DATA + b" Extra sentence here.", "two.txt")
    await ingest_service.delete(first.document.id)
    assert await point_count(qdrant) == second.document.num_chunks


async def test_a_deleted_file_can_be_uploaded_again(
    ingest_service: IngestService,
) -> None:
    first = await ingest(ingest_service)
    await ingest_service.delete(first.document.id)
    again = await ingest(ingest_service)
    assert again.created


async def test_adding_the_same_chunks_twice_does_not_duplicate(
    store: VectorStore, qdrant: AsyncQdrantClient, embedder: FakeEmbedder
) -> None:
    doc_id = uuid4()
    chunks = ["alpha beta", "gamma delta"]
    vectors = embedder.embed_texts(chunks)
    await store.add_chunks(doc_id, "x.txt", chunks, vectors)
    await store.add_chunks(doc_id, "x.txt", chunks, vectors)
    assert await point_count(qdrant) == 2


async def test_mismatched_chunks_and_vectors_are_rejected(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        await store.add_chunks(uuid4(), "x.txt", ["a", "b"], [[0.0] * 384])