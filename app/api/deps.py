from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.services.booking_repository import BookingRepository
from app.services.chat import ChatService
from app.services.health import HealthChecker
from app.services.ingest import IngestService


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_ingest_service(request: Request) -> IngestService:
    service: IngestService = request.app.state.ingest_service
    return service


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    return factory


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


def get_booking_repository(request: Request) -> BookingRepository:
    repository: BookingRepository = request.app.state.booking_repository
    return repository


def get_health_checker(request: Request) -> HealthChecker:
    checker: HealthChecker = request.app.state.health_checker
    return checker


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
SessionFactoryDep = Annotated[
    async_sessionmaker[AsyncSession], Depends(get_session_factory)
]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
BookingRepositoryDep = Annotated[BookingRepository, Depends(get_booking_repository)]
HealthCheckerDep = Annotated[HealthChecker, Depends(get_health_checker)]
