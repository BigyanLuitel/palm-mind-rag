import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 3.0


class HealthReport(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, str]


class HealthChecker:
    """Pings each backing service once, in parallel, with a time limit."""

    def __init__(
        self,
        engine: AsyncEngine,
        redis: Redis,
        qdrant: AsyncQdrantClient,
        timeout: float = CHECK_TIMEOUT_SECONDS,
    ) -> None:
        self._engine = engine
        self._redis = redis
        self._qdrant = qdrant
        self._timeout = timeout

    async def check(self) -> HealthReport:
        probes: dict[str, Callable[[], Awaitable[object]]] = {
            "postgres": self._ping_postgres,
            "redis": self._ping_redis,
            "qdrant": self._ping_qdrant,
        }
        results = await asyncio.gather(
            *(self._run(name, probe) for name, probe in probes.items())
        )
        services = dict(zip(probes, results, strict=True))
        healthy = all(result == "ok" for result in results)
        return HealthReport(status="ok" if healthy else "degraded", services=services)

    async def _run(self, name: str, probe: Callable[[], Awaitable[object]]) -> str:
        try:
            await asyncio.wait_for(probe(), self._timeout)
        except Exception as exc:  # any failure at all means "down"
            logger.warning("Health check failed for %s: %r", name, exc)
            return f"down ({type(exc).__name__})"
        return "ok"

    async def _ping_postgres(self) -> None:
        async with self._engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def _ping_redis(self) -> None:
        await self._redis.ping()

    async def _ping_qdrant(self) -> None:
        await self._qdrant.get_collections()
