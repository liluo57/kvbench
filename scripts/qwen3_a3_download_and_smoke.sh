#!/usr/bin/env bash
# pku14 helper: resume the official Qwen3-8B download, validate every shard,
# and run the A3 smoke test automatically.  It does not modify ragkv.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data1/ly/models/Qwen3-8b}"
GPU_ID="${GPU_ID:-5}"
LOG_FILE="${LOG_FILE:-/data1/ly/Projects/kvbench/outputs/qwen3_a3_pipeline.log}"
mkdir -p "${MODEL_DIR}" "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

TOKEN="$(awk -F= '$1=="export HF_TOKEN"{print substr($0,index($0,"=")+1)}' /data/ly/.bashrc)"
export HF_TOKEN="${TOKEN}"
export HUGGINGFACE_HUB_TOKEN="${TOKEN}"
export HF_ENDPOINT="https://hf-mirror.com"
export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export ALL_PROXY="socks5h://127.0.0.1:7892"

echo "[$(date -Is)] resuming Qwen/Qwen3-8B download"
/data1/ly/envs/cacheblend/bin/python3.10 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="/data1/ly/models/Qwen3-8b",
    resume_download=True,
    max_workers=16,
)
PY

echo "[$(date -Is)] validating safetensors"
/data1/ly/envs/ragkv/bin/python - <<'PY'
from pathlib import Path
from safetensors import safe_open
root = Path("/data1/ly/models/Qwen3-8b")
files = sorted(root.glob("model-*.safetensors"))
if len(files) != 5:
    raise SystemExit(f"expected 5 shards, found {len(files)}")
for path in files:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        print(path.name, "OK", len(handle.keys()), "tensors")
PY

echo "[$(date -Is)] starting A3 Qwen3 smoke on GPU ${GPU_ID}"
cd /data1/ly/Projects/kvbench
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  /data1/ly/envs/cacheblend/bin/python3.10 \
  scripts/qwen3_a3_smoke.py \
    --model "${MODEL_DIR}" \
    --repo-root /data1/ly/Projects/ragkv \
    --python /data1/ly/envs/ragkv/bin/python \
    --gpu "${GPU_ID}" \
    --max-new-tokens 8 \
    --max-model-len 4096

echo "[$(date -Is)] pipeline complete"
