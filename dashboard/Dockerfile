# SlowFast dashboard — CUDA 11.8 + Python 3.10 + Streamlit on port 8501.
#
# The build context is the project root (see dashboard/compose.yml) so that
# bind-mount paths resolve predictably and we can fetch dashboard sources
# from the expected place.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# System deps:
#   - python3.10           → runtime
#   - python3-pip          → bootstrap uv
#   - ffmpeg, libgl1, libglib2.0-0 → pytorchvideo + opencv-headless dependencies
#   - curl                 → HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir "uv>=0.4.0"

WORKDIR /app

# Copy dashboard sources into the image (the training tree itself is bind-
# mounted read-only at runtime, not baked in).
COPY dashboard/pyproject.toml /app/dashboard/pyproject.toml
COPY dashboard/app.py         /app/dashboard/app.py
COPY dashboard/config.py      /app/dashboard/config.py
COPY dashboard/logging_setup.py /app/dashboard/logging_setup.py
COPY dashboard/inference      /app/dashboard/inference
COPY dashboard/ui             /app/dashboard/ui
COPY dashboard/tests          /app/dashboard/tests

# Install dashboard deps (CUDA 11.8 wheels come from pytorch-cu118 index).
WORKDIR /app/dashboard
RUN uv pip install --system .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# The bind-mount makes dashboard/ available at /workspace/dashboard; run
# Streamlit from there so imports like `from config import ...` resolve.
WORKDIR /workspace/dashboard

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
