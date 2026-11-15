import asyncio
from typing import Any

import httpx
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.health import router
from app.services.health import HealthChecker


class BrokenRedis:
    async def ping(self) -> bool:
        raise ConnectionError("secret-host:6379 refused")


class SlowRedis:
    async def ping(self) -> bool:
        await asyncio.sleep(5)
        return True


def checker(
    engine: AsyncEngine, redis: Any, qdrant: AsyncQdrantClient, timeout: float = 3.0
) -> HealthChecker:
    return HealthChecker(engine, redis, qdrant, timeout)


async def test_all_services_up(
    engine: AsyncEngine, redis: Any, qdrant: AsyncQdrantClient
) -> None:
    report = await checker(engine, redis, qdrant).check()
    assert report.status == "ok"
    assert report.services == {"postgres": "ok", "redis": "ok", "qdrant": "ok"}


async def test_one_service_down_is_degraded_and_hides_details(
    engine: AsyncEngine, qdrant: AsyncQdrantClient
) -> None:
    report = await checker(engine, BrokenRedis(), qdrant).check()
    assert report.status == "degraded"
    assert report.services["redis"] == "down (ConnectionError)"
    assert report.services["postgres"] == "ok"
    assert "secret-host" not in report.model_dump_json()


async def test_a_hanging_service_times_out(
    engine: AsyncEngine, qdrant: AsyncQdrantClient
) -> None:
    report = await checker(engine, SlowRedis(), qdrant, timeout=0.05).check()
    assert report.services["redis"] == "down (TimeoutError)"
    assert report.status == "degraded"


async def test_endpoint_returns_200_then_503(
    engine: AsyncEngine, redis: Any, qdrant: AsyncQdrantClient
) -> None:
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        app.state.health_checker = checker(engine, redis, qdrant)
        ok = await client.get("/health")
        assert ok.status_code == 200 and ok.json()["status"] == "ok"

        app.state.health_checker = checker(engine, BrokenRedis(), qdrant)
        down = await client.get("/health")
        assert down.status_code == 503
        assert down.json()["services"]["redis"].startswith("down")