# Serving image with the SageMaker inference contract (GET /ping, POST /invocations on port 8080).
# WRITTEN, NOT BUILT OR RUN in this build: Docker Desktop keeps its disk image on C: by default.
#
# Layout follows SageMaker: model artifacts are mounted at /opt/ml/model (the merged model directory,
# artifacts/merged/llama2-7b-qlora: safetensors + tokenizer). QLORA_ROOT=/opt/ml so every project path
# resolves under it, as on Windows; the repo sits at $QLORA_ROOT/repo, where configs are looked up.
# The image serves the merged model in 4-bit NF4 (configs/serve_container.yaml); no MLflow or Hub access needed.
#
#   docker build -t qlora-cs .
#   docker run --gpus all -p 8080:8080 -v <merged model dir>:/opt/ml/model qlora-cs serve
#
# Note: uv.lock is resolved for Windows only (tool.uv.environments = win32), so the image installs from
# pyproject.toml with the CUDA 12.8 PyTorch index. Re-lock for linux before relying on exact pins.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    QLORA_ROOT=/opt/ml UV_PYTHON_INSTALL_DIR=/opt/ml/tools/python UV_CACHE_DIR=/opt/ml/cache/uv \
    HF_HOME=/opt/ml/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/0.11.6/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /opt/ml/repo
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts/container_entrypoint.sh /usr/local/bin/container_entrypoint.sh

RUN uv python install 3.11 && uv venv --python 3.11 /opt/ml/venv && \
    VIRTUAL_ENV=/opt/ml/venv uv pip install --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 . && \
    mkdir -p /opt/ml/model /opt/ml/tmp /opt/ml/logs /opt/ml/cache && \
    printf 'QLORA_ROOT=/opt/ml\nHF_HOME=/opt/ml/cache/huggingface\nTEMP=/opt/ml/tmp\nTMP=/opt/ml/tmp\n' > /opt/ml/qlora.env && \
    sed -i 's/\r$//' /usr/local/bin/container_entrypoint.sh && chmod +x /usr/local/bin/container_entrypoint.sh

ENV PATH="/opt/ml/venv/bin:${PATH}"
EXPOSE 8080

# SageMaker starts the container as `docker run <image> serve`. That replaces CMD, so the defaults live in the
# entrypoint: `serve` -> `qlora serve -c serve_container.yaml` (0.0.0.0:8080, merged model at /opt/ml/model).
ENTRYPOINT ["/usr/local/bin/container_entrypoint.sh"]
CMD ["serve"]
