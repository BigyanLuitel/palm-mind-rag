from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import IngestServiceDep, SessionFactoryDep, SettingsDep
from app.models.db import ChunkStrategy, Document, DocumentStatus

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    size_bytes: int
    chunk_strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int
    num_chunks: int
    status: DocumentStatus
    created_at: datetime


class UploadOut(DocumentOut):
    duplicate: bool


@router.post("", response_model=UploadOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    response: Response,
    service: IngestServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.SENTENCE,
    chunk_size: Annotated[int | None, Form(ge=100)] = None,
    chunk_overlap: Annotated[int | None, Form(ge=0)] = None,
) -> UploadOut:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap >= size:
        raise HTTPException(
            status_code=422, detail="chunk_overlap must be smaller than chunk_size"
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    filename = Path(file.filename or "upload").name[:255]
    result = await service.ingest(filename, data, strategy, size, overlap)
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return UploadOut(
        **DocumentOut.model_validate(result.document).model_dump(),
        duplicate=not result.created,
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(session_factory: SessionFactoryDep) -> list[Document]:
    async with session_factory() as session:
        query = select(Document).order_by(Document.created_at.desc()).limit(100)
        return list((await session.execute(query)).scalars())


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: UUID, service: IngestServiceDep) -> Response:
    if not await service.delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)