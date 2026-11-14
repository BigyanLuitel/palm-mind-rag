import asyncio
import hashlib
import logging
from collections.abc import Callable
from uuid import UUID

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
    ) -> Document:
        # 1. Cheap checks first: nothing is saved if the file is unusable.
        text = await asyncio.to_thread(extract_text, filename, data)
        chunks = CHUNKERS[strategy](text, chunk_size, chunk_overlap)
        if not chunks:
            raise UnreadableFileError("No text chunks could be created.")

        # 2. Record the document as "processing" before touching Qdrant.
        doc = await self._create_row(
            filename, data, strategy, chunk_size, chunk_overlap
        )

        # 3. Expensive work. On any failure, undo what was written.
        try:
            vectors = await asyncio.to_thread(self._embedder.embed_texts, chunks)
            await self._store.add_chunks(doc.id, filename, chunks, vectors)
            return await self._finish(doc.id, DocumentStatus.READY, len(chunks))
        except Exception:
            await self._cleanup(doc.id)
            raise

    async def _create_row(
        self,
        filename: str,
        data: bytes,
        strategy: ChunkStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        doc = Document(
            filename=filename,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
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