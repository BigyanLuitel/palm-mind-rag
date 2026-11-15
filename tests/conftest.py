from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis.aioredis
import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.api.bookings import router as bookings_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.errors import register_error_handlers
from app.core.database import create_session_factory, init_db
from app.services.booking_flow import BookingFlow
from app.services.booking_repository import BookingRepository
from app.services.booking_state import BookingStateStore
from app.services.chat import ChatService
from app.services.chat_memory import ChatMemory
from app.services.ingest import IngestService
from app.services.rag import RagService
from app.services.vector_store import VectorStore
from tests.fakes import (
    COLLECTION,
    DIM,
    NOW,
    FakeBookingRepository,
    FakeEmbedder,
    ScriptedLLM,
    make_settings,
)


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def states(redis: Redis) -> BookingStateStore:
    return BookingStateStore(redis, ttl_seconds=1_800)


@pytest.fixture
def repository() -> FakeBookingRepository:
    return FakeBookingRepository()


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """A real SQL database (SQLite in a temp file) with the app's tables."""
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(db_engine)
    yield db_engine
    await db_engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return create_session_factory(engine)


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(":memory:")
    yield client
    await client.close()


@pytest.fixture
async def store(qdrant: AsyncQdrantClient) -> VectorStore:
    vector_store = VectorStore(qdrant, COLLECTION, DIM)
    await vector_store.ensure_collection()
    return vector_store


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def ingest_service(
    session_factory: async_sessionmaker[AsyncSession],
    store: VectorStore,
    embedder: FakeEmbedder,
) -> IngestService:
    return IngestService(session_factory, store, embedder)


@pytest.fixture
def api_app(
    session_factory: async_sessionmaker[AsyncSession],
    ingest_service: IngestService,
    store: VectorStore,
    embedder: FakeEmbedder,
    llm: ScriptedLLM,
    redis: Redis,
) -> FastAPI:
    """The real routers and services on fake backends (no startup, no model)."""
    app = FastAPI()
    register_error_handlers(app)
    for router in (documents_router, chat_router, bookings_router):
        app.include_router(router)
    bookings = BookingRepository(session_factory)
    flow = BookingFlow(
        llm,  # type: ignore[arg-type]
        BookingStateStore(redis, 1_800),
        bookings,
        clock=lambda: NOW,
    )
    rag = RagService(embedder, store, llm, 5, 0.25, 0.2)  # type: ignore[arg-type]
    app.state.settings = make_settings()
    app.state.session_factory = session_factory
    app.state.ingest_service = ingest_service
    app.state.chat_service = ChatService(ChatMemory(redis, 10, 3_600), rag, flow)
    app.state.booking_repository = bookings
    return app


@pytest.fixture
async def client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c