from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
    ) -> None:
        self._model_name = model_name
        self.dim = dim
        self._model: TextEmbedding | None = None

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]