FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY requirements.txt .
# hdbscan ships no Linux wheel, so it compiles; the toolchain is removed afterwards.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Bake the embedding model, embeddings and topic results into the image so the
# container starts in seconds and needs no network access at runtime.
RUN python -c "from config import get_settings; from pipeline import build_index; build_index(get_settings())"

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
