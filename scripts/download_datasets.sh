#!/bin/bash
# Download all 7 PPE datasets for Munin v4
# Usage: ./scripts/download_datasets.sh [/scratch/datasets/raw]
set -euo pipefail

DATA_DIR="${1:-/scratch/datasets/raw}"
mkdir -p "$DATA_DIR"

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# ─────────────────────────────────────────────────────────────
# 1. Youcefs — Construction PPE (Roboflow)
# ─────────────────────────────────────────────────────────────
download_youcefs() {
    local dest="$DATA_DIR/youcefs"
    if [[ -d "$dest/images" ]]; then
        log "SKIP: Youcefs already exists at $dest"
        return
    fi
    log "Downloading Youcefs Construction PPE..."
    mkdir -p "$dest"

    if command -v roboflow &>/dev/null; then
        roboflow download \
            --workspace youcefs \
            --project construction-ppe \
            --version 1 \
            --format yolov8 \
            --location "$dest"
    else
        log "WARN: roboflow CLI not found. Install: pip install roboflow"
        log "WARN: Skipping Youcefs. Run dataset_builder.py with Roboflow API key."
    fi
}

# ─────────────────────────────────────────────────────────────
# 2. keremberke — PPE Detection (HuggingFace)
# ─────────────────────────────────────────────────────────────
download_keremberke() {
    local dest="$DATA_DIR/keremberke"
    if [[ -d "$dest/images" ]]; then
        log "SKIP: keremberke already exists at $dest"
        return
    fi
    log "Downloading keremberke PPE Detection (HuggingFace)..."

    python3 -c "
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

dest = Path('$dest')
img_dir = dest / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset('keremberke/ppe-detection', 'yolov8', split='train', trust_remote_code=True)
for i, item in enumerate(tqdm(ds, desc='keremberke')):
    img = item.get('image')
    if img:
        img.save(img_dir / f'img_{i:06d}.jpg', format='JPEG')
print(f'Downloaded {i+1} images')
"
}

# ─────────────────────────────────────────────────────────────
# 3. skcet — PPE Detection (Roboflow)
# ─────────────────────────────────────────────────────────────
download_skcet() {
    local dest="$DATA_DIR/skcet"
    if [[ -d "$dest/images" ]]; then
        log "SKIP: skcet already exists at $dest"
        return
    fi
    log "Downloading skcet PPE Detection..."
    mkdir -p "$dest"

    if command -v roboflow &>/dev/null; then
        roboflow download \
            --workspace skcet \
            --project ppe-detection-v9rfk \
            --version 1 \
            --format yolov8 \
            --location "$dest"
    else
        log "WARN: roboflow CLI not found. Skipping skcet."
    fi
}

# ─────────────────────────────────────────────────────────────
# 4. SHWD — Safety Helmet/Wearing Detection (GitHub)
# ─────────────────────────────────────────────────────────────
download_shwd() {
    local dest="$DATA_DIR/shwd"
    if [[ -d "$dest/.git" ]]; then
        log "SKIP: SHWD already cloned at $dest"
        return
    fi
    log "Cloning SHWD from GitHub..."
    git clone --depth 1 https://github.com/ggiscan/SHWD.git "$dest"
}

# ─────────────────────────────────────────────────────────────
# 5. Construction 2 (Roboflow)
# ─────────────────────────────────────────────────────────────
download_construction2() {
    local dest="$DATA_DIR/construction2"
    if [[ -d "$dest/images" ]]; then
        log "SKIP: construction2 already exists at $dest"
        return
    fi
    log "Downloading Construction 2 PPE Dataset..."
    mkdir -p "$dest"

    if command -v roboflow &>/dev/null; then
        roboflow download \
            --workspace construction \
            --project ppe-construction-2 \
            --version 1 \
            --format yolov8 \
            --location "$dest"
    else
        log "WARN: roboflow CLI not found. Skipping construction2."
    fi
}

# ─────────────────────────────────────────────────────────────
# 6. VoxDroid — PPE Detection (GitHub)
# ─────────────────────────────────────────────────────────────
download_voxdroid() {
    local dest="$DATA_DIR/voxdroid"
    if [[ -d "$dest/.git" ]]; then
        log "SKIP: VoxDroid already cloned at $dest"
        return
    fi
    log "Cloning VoxDroid PPE Detection from GitHub..."
    git clone --depth 1 https://github.com/VoxDroid/PPE-Detection.git "$dest"
}

# ─────────────────────────────────────────────────────────────
# 7. keremberke small (HuggingFace)
# ─────────────────────────────────────────────────────────────
download_keremberke_small() {
    local dest="$DATA_DIR/keremberke_small"
    if [[ -d "$dest/images" ]]; then
        log "SKIP: keremberke_small already exists at $dest"
        return
    fi
    log "Downloading keremberke small PPE Detection (HuggingFace)..."

    python3 -c "
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

dest = Path('$dest')
img_dir = dest / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset('keremberke/ppe-detection-small', 'yolov8', split='train', trust_remote_code=True)
for i, item in enumerate(tqdm(ds, desc='keremberke_small')):
    img = item.get('image')
    if img:
        img.save(img_dir / f'img_{i:06d}.jpg', format='JPEG')
print(f'Downloaded {i+1} images')
"
}

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
log "=== Munin v4: Downloading all 7 PPE datasets ==="
log "Target directory: $DATA_DIR"
echo ""

download_youcefs
echo ""
download_keremberke
echo ""
download_skcet
echo ""
download_shwd
echo ""
download_construction2
echo ""
download_voxdroid
echo ""
download_keremberke_small
echo ""

log "=== All downloads complete ==="
log "Run dataset_builder.py to unify:
    python scripts/dataset_builder.py --output /scratch/datasets/munin-v4"
