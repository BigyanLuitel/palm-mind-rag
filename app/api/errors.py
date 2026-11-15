import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.services.extractor import UnreadableFileError, UnsupportedFileTypeError
from app.services.llm import LLMUnavailableError

logger = logging.getLogger(__name__)


async def _unsupported_type(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(exc)})


async def _unreadable_file(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _service_unavailable(_: Request, exc: Exception) -> JSONResponse:
    logger.error("A backing service failed", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "A required service is temporarily unavailable."},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(UnsupportedFileTypeError, _unsupported_type)
    app.add_exception_handler(UnreadableFileError, _unreadable_file)
    for exc_type in (
        LLMUnavailableError,
        ResponseHandlingException,
        UnexpectedResponse,
        SQLAlchemyError,
        RedisError,
        ConnectionError,
    ):
        app.add_exception_handler(exc_type, _service_unavailable)