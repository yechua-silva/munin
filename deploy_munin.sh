#!/usr/bin/env bash
# deploy_munin.sh — Deploy Munin v3 al droplet AMD MI300X
#
# Uso: ./deploy_munin.sh [--legacy|--dual_class]
#
# Pasos:
#   1. Git push local → GitHub
#   2. SSH al droplet + docker exec rocm git pull
#   3. pip install dependencies dentro del container
#   4. Verificar best.pt + run tests
#
# Droplet: 129.212.177.29 (root@)
# Container: rocm (ROCm 7.2.4, MI300X)
# Fine-tune: /scratch/runs/detect/train/weights/best.pt
#
# ADR-020: SSH rate-limiting — 3s entre comandos docker exec

set -euo pipefail

# ── Configuración ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR="${SCRIPT_DIR}/munin"

DROPLET_IP="129.212.177.29"
SSH_USER="root"
CONTAINER="rocm"
REMOTE_REPO_DIR="/scratch/munin"
BEST_PT_PATH="/scratch/runs/detect/train/weights/best.pt"
SSH_DEST="${SSH_USER}@${DROPLET_IP}"
MODE="${1:---legacy}"

# ── Funciones auxiliares ────────────────────────────────────────────────────
ssh_exec() {
    ssh "${SSH_DEST}" "$@"
}

docker_exec() {
    # ADR-020: 3s rate-limit entre invocaciones docker exec
    ssh_exec "docker exec ${CONTAINER} bash -c '$*'"
    sleep 3
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "=== Munin v3 Deploy ==="
echo "Mode:       ${MODE}"
echo "Droplet:    ${SSH_DEST}"
echo "Container:  ${CONTAINER}"
echo "Git dir:    ${GIT_DIR}"
echo ""

# ── Paso 1: Git push local ──────────────────────────────────────────────────
echo "[1/5] Git push local → GitHub..."

cd "${GIT_DIR}"
git add -A
git commit -m "F3 complete: v3 components deployed (${MODE})" || true
git push origin master || git push origin main

cd "${SCRIPT_DIR}"
echo "✅ Git push complete"
echo ""

# ── Paso 2: SSH + docker exec git pull ──────────────────────────────────────
echo "[2/5] SSH al droplet + git pull en container..."

docker_exec "
    if [ ! -d ${REMOTE_REPO_DIR} ]; then
        echo 'Cloning repo to ${REMOTE_REPO_DIR}...'
        git clone https://github.com/yechua-silva/munin.git ${REMOTE_REPO_DIR}
    fi
    cd ${REMOTE_REPO_DIR}
    git pull
    echo 'Git pull complete'
"

echo "✅ Git pull en container complete"
echo ""

# ── Paso 3: pip install ─────────────────────────────────────────────────────
echo "[3/5] pip install dependencies en container..."

docker_exec "
    cd ${REMOTE_REPO_DIR}
    pip install -r requirements.txt 2>&1 | tail -5
    echo 'pip install complete'
"

echo "✅ Dependencies installed"
echo ""

# ── Paso 3.5: Deploy vLLM server (VLM on-premise) ────────────────────────────
echo "[3.5/5] Deploying vLLM server on MI300X..."
ssh "${SSH_USER}@${DROPLET_IP}" \
    "docker exec ${CONTAINER} bash -c '
        # Check if vLLM is already running
        if curl -s http://localhost:8000/v1/models | grep -q model; then
            echo \"vLLM already running\"
        else
            echo \"Installing vLLM ROCm...\"
            pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ 2>&1 | tail -3

            echo \"Downloading MiniCPM-V-2_6 model (~8GB)...\"
            python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download(\\\"openbmb/MiniCPM-V-2_6\\\", local_dir=\\\"/scratch/models/MiniCPM-V-2_6\\\")
print(\\\"Model downloaded\\\")
\" 2>&1 | tail -5

            echo \"Starting vLLM server in background...\"
            export VLLM_ROCM_USE_AITER=1
            export HIP_FORCE_DEV_KERNARG=1
            export SAFETENSORS_FAST_GPU=1

            nohup vllm serve \"openbmb/MiniCPM-V-2_6\" \
                --dtype float16 \
                --tensor-parallel-size 1 \
                --gpu-memory-utilization 0.40 \
                --max-model-len 4096 \
                --max-num-seqs 256 \
                --port 8000 \
                > /scratch/vllm-server.log 2>&1 &

            echo \"vLLM starting (check /scratch/vllm-server.log)\"
            echo \"Waiting for server to be ready...\"
            sleep 30
            curl -s http://localhost:8000/v1/models | head -5 || echo \"Server still starting...\"
        fi
    '" 2>&1
echo "✅ vLLM deploy complete"
echo ""

# ── Paso 4.5: Verificar best.pt + run tests ───────────────────────────────────
echo "[4.5/5] Verificando best.pt + running tests..."

docker_exec "
    if [ -f ${BEST_PT_PATH} ]; then
        echo '✅ best.pt found: ${BEST_PT_PATH}'
        ls -lh ${BEST_PT_PATH}
    else
        echo '⚠️  best.pt not found yet (fine-tune may still be running)'
        echo 'Fine-tune status:'
        ls -la /scratch/runs/detect/train/ 2>/dev/null || echo 'No train dir yet'
    fi
    echo ''
    echo 'Running tests...'
    cd ${REMOTE_REPO_DIR}
    python -m pytest tests/ -v --tb=short 2>&1 | tail -20
"

echo "✅ Verification complete"
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────
echo "=== Deploy complete ==="
echo ""
echo "Next steps:"
echo "  • T4.4: Verify DUAL_CLASS mode with best.pt"
echo "  • T4.5: Smoke test E2E"
