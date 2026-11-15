from typing import Any

import httpx
from fastapi import FastAPI

from app.api.deps import get_ingest_service

POLICY = (
    b"Refunds are accepted within 30 days of purchase. "
    b"Shipping takes three to five business days."
)
MAX_BYTES = 10 * 1024 * 1024


async def upload(
    client: httpx.AsyncClient,
    name: str = "policy.txt",
    content: bytes = POLICY,
    **form: Any,
) -> httpx.Response:
    return await client.post(
        "/api/v1/documents",
        files={"file": (name, content, "application/octet-stream")},
        data={key: str(value) for key, value in form.items()},
    )


async def test_upload_creates_a_ready_document(client: httpx.AsyncClient) -> None:
    response = await upload(client)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "policy.txt"
    assert body["status"] == "ready"
    assert body["duplicate"] is False
    assert body["num_chunks"] >= 1
    assert body["chunk_strategy"] == "sentence"


async def test_same_file_again_returns_200_and_duplicate(
    client: httpx.AsyncClient,
) -> None:
    first = (await upload(client)).json()
    response = await upload(client)
    assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert response.json()["id"] == first["id"]


async def test_fixed_strategy_with_custom_sizes(client: httpx.AsyncClient) -> None:
    response = await upload(client, strategy="fixed", chunk_size=100, chunk_overlap=20)
    assert response.status_code == 201
    body = response.json()
    assert body["chunk_strategy"] == "fixed"
    assert (body["chunk_size"], body["chunk_overlap"]) == (100, 20)


async def test_pdf_and_txt_only(client: httpx.AsyncClient) -> None:
    response = await upload(client, name="report.docx", content=b"hello")
    assert response.status_code == 415


async def test_empty_and_binary_text_files_are_422(client: httpx.AsyncClient) -> None:
    assert (await upload(client, content=b"  \n ")).status_code == 422
    assert (await upload(client, content=b"\xff\xfe\x00bad")).status_code == 422


async def test_overlap_not_smaller_than_size_is_422(client: httpx.AsyncClient) -> None:
    response = await upload(client, chunk_size=100, chunk_overlap=100)
    assert response.status_code == 422


async def test_chunk_size_below_100_is_422(client: httpx.AsyncClient) -> None:
    assert (await upload(client, chunk_size=50)).status_code == 422


async def test_unknown_strategy_is_422(client: httpx.AsyncClient) -> None:
    assert (await upload(client, strategy="magic")).status_code == 422


async def test_missing_file_is_422(client: httpx.AsyncClient) -> None:
    assert (await client.post("/api/v1/documents")).status_code == 422


async def test_file_over_10_mb_is_413(client: httpx.AsyncClient) -> None:
    response = await upload(client, content=b"a" * (MAX_BYTES + 1))
    assert response.status_code == 413


async def test_list_shows_uploaded_documents(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/documents")).json() == []
    uploaded = (await upload(client)).json()
    listed = (await client.get("/api/v1/documents")).json()
    assert [d["id"] for d in listed] == [uploaded["id"]]


async def test_delete_then_gone(client: httpx.AsyncClient) -> None:
    doc_id = (await upload(client)).json()["id"]
    assert (await client.delete(f"/api/v1/documents/{doc_id}")).status_code == 204
    assert (await client.get("/api/v1/documents")).json() == []
    assert (await client.delete(f"/api/v1/documents/{doc_id}")).status_code == 404


async def test_delete_with_a_bad_id_is_422(client: httpx.AsyncClient) -> None:
    assert (await client.delete("/api/v1/documents/not-a-uuid")).status_code == 422


async def test_a_backing_service_failure_becomes_503(
    client: httpx.AsyncClient, api_app: FastAPI
) -> None:
    class Exploding:
        async def ingest(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("database is down")

    api_app.dependency_overrides[get_ingest_service] = lambda: Exploding()
    response = await upload(client)
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
    assert "database is down" not in response.text  # internals stay hidden
