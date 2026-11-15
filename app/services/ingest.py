import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.db import ChunkStrategy, Document, DocumentStatus
from app.services.chunker import fixed_chunks, sentence_chunks
from app.services.embedder import Embedder
from app.services.extractor import UnreadableFileError, extract_text
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

CHUNKERS: dict[ChunkStrategy, Callable[[str, int, int], list[str]]] = {
    ChunkStrategy.FIXED: fixed_chunks,
    ChunkStrategy.SENTENCE: sentence_chunks,
}

LIVE_STATUSES = (DocumentStatus.PROCESSING, DocumentStatus.READY)


@dataclass(frozen=True)
class IngestResult:
    document: Document
    created: bool  # False means this exact file was already uploaded


class IngestService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        self._session_factory = session_factory
        self._store = vector_store
        self._embedder = embedder

    async def ingest(
        self,
        filename: str,
        data: bytes,
        strategy: ChunkStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> IngestResult:
        # 1. Same file already uploaded? Return it and skip all the work.
        sha256 = hashlib.sha256(data).hexdigest()
        existing = await self._find_live(sha256)
        if existing is not None:
            return IngestResult(existing, created=False)

        # 2. Cheap checks next: nothing is saved if the file is unusable.
        text = await asyncio.to_thread(extract_text, filename, data)
        chunks = CHUNKERS[strategy](text, chunk_size, chunk_overlap)
        if not chunks:
            raise UnreadableFileError("No text chunks could be created.")

        # 3. Record the document as "processing" before touching Qdrant.
        try:
            doc = await self._create_row(
                filename, len(data), sha256, strategy, chunk_size, chunk_overlap
            )
        except IntegrityError:
            # Another request uploaded the same file a moment ago and won the race.
            existing = await self._find_live(sha256)
            if existing is None:
                raise
            return IngestResult(existing, created=False)

        # 4. Expensive work. On any failure, undo what was written.
        try:
            vectors = await asyncio.to_thread(self._embedder.embed_texts, chunks)
            await self._store.add_chunks(doc.id, filename, chunks, vectors)
            ready = await self._finish(doc.id, DocumentStatus.READY, len(chunks))
            return IngestResult(ready, created=True)
        except Exception:
            await self._cleanup(doc.id)
            raise

    async def delete(self, doc_id: UUID) -> bool:
        """Remove a document's vectors and row. Returns False if it does not exist."""
        async with self._session_factory() as session:
            doc = await session.get(Document, doc_id)
            if doc is None:
                return False
            await self._store.delete_document(doc_id)  # vectors first, safe to retry
            await session.delete(doc)
            await session.commit()
            return True

    async def _find_live(self, sha256: str) -> Document | None:
        query = select(Document).where(
            Document.sha256 == sha256, Document.status.in_(LIVE_STATUSES)
        )
        async with self._session_factory() as session:
            return (await session.execute(query)).scalars().first()

    async def _create_row(
        self,
        filename: str,
        size_bytes: int,
        sha256: str,
        strategy: ChunkStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        doc = Document(
            filename=filename,
            size_bytes=size_bytes,
            sha256=sha256,
            chunk_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        async with self._session_factory() as session:
            session.add(doc)
            await session.commit()
        return doc

    async def _finish(
        self, doc_id: UUID, status: DocumentStatus, num_chunks: int = 0
    ) -> Document:
        async with self._session_factory() as session:
            doc = await session.get(Document, doc_id)
            if doc is None:
                raise RuntimeError(f"Document {doc_id} disappeared during ingest")
            doc.status = status
            doc.num_chunks = num_chunks
            await session.commit()
            return doc

    async def _cleanup(self, doc_id: UUID) -> None:
        try:
            await self._store.delete_document(doc_id)
        except Exception:
            logger.exception("Could not delete vectors for document %s", doc_id)
        try:
            await self._finish(doc_id, DocumentStatus.FAILED)
        except Exception:
            logger.exception("Could not mark document %s as failed", doc_id)
