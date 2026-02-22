FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsox-dev ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Copy the Qwen3-TTS source
COPY . /app/

# Install Qwen3-TTS + service deps + ASR
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -e ".[service,service-asr]" && \
    pip install --no-cache-dir runpod && \
    pip install --no-cache-dir flash-attn --no-build-isolation || true

# Pre-download the model at build time so cold-start is fast
RUN python -c "\
    from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', local_dir='/app/models/Qwen3-TTS-12Hz-0.6B-Base')"

# Copy the serverless handler
COPY handler.py /app/handler.py

ENV QWEN3_TTS_CHECKPOINT=/app/models/Qwen3-TTS-12Hz-0.6B-Base
ENV QWEN3_TTS_DEVICE=cuda:0
ENV QWEN3_TTS_DTYPE=bfloat16
ENV QWEN3_TTS_FLASH_ATTENTION=true
ENV QWEN3_TTS_AUTO_TRANSCRIBE=true
ENV QWEN3_TTS_ASR_MODEL=base
ENV QWEN3_TTS_ASR_DEVICE=cuda
ENV QWEN3_TTS_ASR_COMPUTE_TYPE=float16
ENV VOICE_PROFILE_DIR=/app/voice_profiles

CMD ["python", "-u", "/app/handler.py"]
