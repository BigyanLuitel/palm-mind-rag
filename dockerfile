FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

# onnxruntime (used by fastembed) needs the OpenMP runtime, which slim lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing app code does not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image: no download on first start.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

RUN useradd --create-home --uid 1000 app && chown -R app:app /opt/fastembed
COPY --chown=app:app app ./app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]