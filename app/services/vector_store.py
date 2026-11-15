from uuid import UUID, uuid5
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings

BATCH_SIZE = 100
class RetrievedChunk(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float

def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dim: int) -> None:
        self._client = client
        self._collection = collection
        self._dim = dim

    async def ensure_collection(self) -> None:
        """Create the collection (and a doc_id index) if it does not exist yet."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dim, distance=models.Distance.COSINE
            ),
        )
        await self._client.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    async def add_chunks(
        self,
        doc_id: UUID,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = [
            models.PointStruct(
                id=str(uuid5(doc_id, str(index))),
                vector=vector,
                payload={
                    "doc_id": str(doc_id),
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        for start in range(0, len(points), BATCH_SIZE):
            await self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + BATCH_SIZE],
            )

    async def delete_document(self, doc_id: UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=str(doc_id))
                        )
                    ]
                )
            ),
        )

    async def search(
        self, vector: list[float], limit: int, min_score: float
    ) -> list[RetrievedChunk]:
        """Return the closest chunks, best first, ignoring weak matches."""
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            score_threshold=min_score,
            with_payload=True,
        )
        return [
            RetrievedChunk.model_validate({**(point.payload or {}), "score": point.score})
            for point in response.points
        ]