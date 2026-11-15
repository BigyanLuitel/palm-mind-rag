import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.bookings import router as bookings_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.database import create_engine, create_session_factory, init_db
from app.services.booking_flow import BookingFlow
from app.services.booking_repository import BookingRepository
from app.services.booking_state import BookingStateStore
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemory, create_redis_client
from app.services.embedder import FastEmbedder
from app.services.health import HealthChecker
from app.services.ingest import IngestService
from app.services.llm import create_groq_client
from app.services.rag import RagService
from app.services.vector_store import VectorStore, create_qdrant_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    engine = create_engine(settings.database_url)
    await init_db(engine)
    session_factory = create_session_factory(engine)

    qdrant = create_qdrant_client(settings)
    embedder = FastEmbedder(settings.embedding_model, settings.embedding_dim)
    await asyncio.to_thread(embedder.embed_query, "warm up")  # load the model now
    store = VectorStore(qdrant, settings.qdrant_collection, settings.embedding_dim)
    await store.ensure_collection()

    redis = create_redis_client(settings)
    memory = ChatMemory(redis, settings.chat_history_length, settings.chat_ttl_seconds)
    llm = create_groq_client(settings)
    rag = RagService(
        embedder,
        store,
        llm,
        settings.retrieval_top_k,
        settings.retrieval_min_score,
        settings.retrieval_score_margin,
    )

    booking_repository = BookingRepository(session_factory)
    booking_flow = BookingFlow(
        llm, BookingStateStore(redis, settings.booking_ttl_seconds), booking_repository
    )

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.ingest_service = IngestService(session_factory, store, embedder)
    app.state.chat_service = ChatService(memory, rag, booking_flow)
    app.state.booking_repository = booking_repository
    app.state.health_checker = HealthChecker(engine, redis, qdrant)
    try:
        yield
    finally:
        await redis.aclose()
        await qdrant.close()
        await engine.dispose()


app = FastAPI(title="Palm Mind RAG", lifespan=lifespan)
register_error_handlers(app)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(bookings_router)