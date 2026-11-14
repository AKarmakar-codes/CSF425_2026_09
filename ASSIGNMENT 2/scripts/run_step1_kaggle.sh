#!/usr/bin/env bash
# ==============================================================================
# CS F425 Deep Learning - Assignment II
# Step 1 Execution Script for Kaggle GPU
# ==============================================================================
set -e

echo "=== [Step 1] Starting Source Data Distillation & Preparation ==="

# 1. Environment & Dependencies Setup
echo "Installing required dependencies..."
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python-headless pillow scipy matplotlib tqdm

# 2. Check GPU Availability
python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 3. Locate Source GraphPCB Dataset
SOURCE_DIR=""

echo "Searching for GraphPCB (Graph-W and Graph-F) in /kaggle/input..."
if [ -d "/kaggle/input" ]; then
    # Search entire /kaggle/input tree without depth restriction (case-insensitive)
    for cand in $(find /kaggle/input -type d -iname "*graph-w*" 2>/dev/null); do
        parent="$(dirname "$cand")"
        if [ -d "$parent/Graph-F" ] || [ -d "$parent/graph-f" ]; then
            SOURCE_DIR="$parent"
            break
        fi
    done

    # If not found as extracted directory, check for zip archives (e.g. GraphPCB.zip)
    if [ -z "$SOURCE_DIR" ]; then
        for z in $(find /kaggle/input -type f -iname "*graph*.zip" -o -iname "*pcb*.zip" 2>/dev/null); do
            echo "Found dataset archive: $z. Extracting to /kaggle/working/data/GraphPCB..."
            mkdir -p /kaggle/working/data/GraphPCB
            unzip -q -o "$z" -d /kaggle/working/data/GraphPCB
            for cand in $(find /kaggle/working/data/GraphPCB -type d -iname "*graph-w*" 2>/dev/null); do
                parent="$(dirname "$cand")"
                if [ -d "$parent/Graph-F" ] || [ -d "$parent/graph-f" ]; then
                    SOURCE_DIR="$parent"
                    break 2
                fi
            done
        done
    fi
fi

# Fallback check across common specific paths
if [ -z "$SOURCE_DIR" ]; then
    POSSIBLE_PATHS=(
        "/kaggle/input/the-graphpcb-dataset/GraphPCB"
        "/kaggle/input/the-graphpcb-dataset"
        "/kaggle/input/datasets/the-graphpcb-dataset/GraphPCB"
        "/kaggle/input/datasets/the-graphpcb-dataset"
        "/kaggle/input/graphpcb/GraphPCB"
        "/kaggle/input/graphpcb"
        "data/GraphPCB"
        "../data/GraphPCB"
        "../../data/GraphPCB"
    )
    for p in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$p/Graph-W" ] && [ -d "$p/Graph-F" ]; then
            SOURCE_DIR="$p"
            break
        fi
    done
fi

if [ -z "$SOURCE_DIR" ]; then
    echo "[Error] Could not locate GraphPCB dataset (containing Graph-W and Graph-F)."
    echo "Files found in /kaggle/input:"
    find /kaggle/input -maxdepth 5 2>/dev/null || ls -la /kaggle/input 2>/dev/null
    exit 1
fi
echo "Using GraphPCB dataset from: $SOURCE_DIR"

# 4. Download SAM vit_h Checkpoint if not already present
mkdir -p checkpoints
SAM_CKPT="checkpoints/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CKPT" ] || [ $(stat -c%s "$SAM_CKPT" 2>/dev/null || stat -f%z "$SAM_CKPT") -lt 1000000000 ]; then
    echo "Downloading SAM vit_h weights (2.56 GB)..."
    wget -c --progress=bar:force:noscroll https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$SAM_CKPT"
fi
echo "SAM checkpoint ready: $SAM_CKPT"

# 5. Run Step 1 Part 1: Offline SAM Geometry Distillation
echo ""
echo "=== Step 1.1: Running SAM Geometry Distillation on Graph-W and Graph-F ==="
python3 -u scripts/offline_teacher_source.py \
    --source "$SOURCE_DIR" \
    --sam-ckpt "$SAM_CKPT" \
    --sam-type vit_h \
    --out "data/distilled_source" \
    --device auto \
    --batch-size 32 \
    --mask-nms-thresh 0.50

# 6. Run Step 1 Part 2: Board-Grouped Split & 1024x1024 Tiling
echo ""
echo "=== Step 1.2: Running Board-Grouped Split (131/21) & 1024x1024 Tiling ==="
python3 -u src/semantic/prepare_source.py \
    --source "data/distilled_source" \
    --out "data/prepared_tiles" \
    --seed 42 \
    --tile 1024 \
    --overlap 256 \
    --target-val-boards 21

echo ""
echo "=== [Step 1] Completed Successfully! ==="
echo "Distilled source saved to: data/distilled_source"
echo "Prepared tiles saved to:   data/prepared_tiles"
