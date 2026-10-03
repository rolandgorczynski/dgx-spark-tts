# Local image for NVIDIA DGX Spark (Grace Blackwell, linux/arm64, sm_121).
# Base tag verified on Docker Hub as a linux/arm64 SBSA image (NVARCH=sbsa),
# which is the Grace server ABI, not the Jetson aarch64 ABI:
#   nvidia/cuda:13.0.3-runtime-ubuntu22.04
# PyTorch wheels verified 2026-10-03 on the cu130 index (cp310 = Ubuntu 22.04 python3):
#   torch-2.10.0+cu130-cp310-cp310-manylinux_2_28_aarch64.whl
#   torchaudio-2.10.0+cu130-cp310-cp310-manylinux_2_28_aarch64.whl
# There is no published application image. This Dockerfile is the build.
# Build it on the Spark itself (native arm64). Do not build on x86 and copy.

FROM nvidia/cuda:13.0.3-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHATTERBOX_T3_MODEL=v3 \
    TTS_LANGUAGE=de \
    PORT=8004

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# 1) Python deps (may pull a generic torch as a transitive dependency).
# 2) Official Chatterbox at the commit that contains Multilingual V3, without
#    its torch==2.6.0 pin.
# 3) Re-install the CUDA 13 aarch64 wheels last so they win.
ARG CHATTERBOX_COMMIT=5de7a54aa4e5e2baadb0182dde554908b48b85c2
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt \
    && python3 -m pip install --no-cache-dir --no-deps \
        "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_COMMIT}" \
    && python3 -m pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.10.0+cu130 \
        torchaudio==2.10.0+cu130 \
    && python3 -c "import torch; v=torch.__version__; c=torch.version.cuda or ''; assert v.startswith('2.10.0'), v; assert c.startswith('13'), (v, c); import chatterbox.mtl_tts; print('torch', v, 'cuda', c)"

COPY server.py /app/server.py

RUN mkdir -p /models/huggingface /voices

EXPOSE 8004

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8004"]
