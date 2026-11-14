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
    # 1. Search for any directory containing Graph-W or graph-w
    for cand in $(find /kaggle/input -type d \( -iname "Graph-W" -o -iname "graph-w" \) 2>/dev/null); do
        parent="$(dirname "$cand")"
        if [ -d "$parent/Graph-F" ] || [ -d "$parent/graph-f" ]; then
            SOURCE_DIR="$parent"
            break
        fi
        if [ -d "$cand/Graph-F" ] || [ -d "$cand/graph-f" ]; then
            SOURCE_DIR="$cand"
            break
        fi
    done

    # 2. Check parent directories or subdirectories for GraphPCB
    if [ -z "$SOURCE_DIR" ]; then
        for cand in $(find /kaggle/input -type d \( -iname "*graph*" -o -iname "*pcb*" \) 2>/dev/null); do
            if ([ -d "$cand/Graph-W" ] || [ -d "$cand/graph-w" ]) && ([ -d "$cand/Graph-F" ] || [ -d "$cand/graph-f" ]); then
                SOURCE_DIR="$cand"
                break
            fi
        done
    fi

    # 3. Check for any zip or tar archive in /kaggle/input and extract to /tmp/GraphPCB (saving /kaggle/working quota)
    if [ -z "$SOURCE_DIR" ]; then
        for z in $(find /kaggle/input -type f \( -iname "*.zip" -o -iname "*.tar*" -o -iname "*.tgz" \) 2>/dev/null); do
            echo "Found dataset archive: $z"
            echo "Extracting archive to /tmp/GraphPCB (outside /kaggle/working)..."
            mkdir -p /tmp/GraphPCB
            if [[ "$z" == *.zip ]]; then
                unzip -o "$z" -d /tmp/GraphPCB
            else
                tar -xf "$z" -C /tmp/GraphPCB
            fi
            for cand in $(find /tmp/GraphPCB -type d \( -iname "Graph-W" -o -iname "graph-w" \) 2>/dev/null); do
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
        if ([ -d "$p/Graph-W" ] || [ -d "$p/graph-w" ]) && ([ -d "$p/Graph-F" ] || [ -d "$p/graph-f" ]); then
            SOURCE_DIR="$p"
            break
        fi
    done
fi

if [ -z "$SOURCE_DIR" ]; then
    echo "[Error] Could not locate GraphPCB dataset (containing Graph-W and Graph-F)."
    echo "Directory listing of /kaggle/input (up to 4 levels deep):"
    find /kaggle/input -maxdepth 4 2>/dev/null
    exit 1
fi
echo "Using GraphPCB dataset from: $SOURCE_DIR"

# 4. Download SAM vit_h Checkpoint to /tmp (avoids 20 GB /kaggle/working disk cap)
mkdir -p /tmp/checkpoints
SAM_CKPT="/tmp/checkpoints/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CKPT" ] || [ $(stat -c%s "$SAM_CKPT" 2>/dev/null || stat -f%z "$SAM_CKPT") -lt 1000000000 ]; then
    echo "Downloading SAM vit_h weights (2.56 GB) to /tmp..."
    wget -c --progress=bar:force:noscroll https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$SAM_CKPT"
fi
echo "SAM checkpoint ready: $SAM_CKPT"

# 5. Run Step 1 Part 1: Offline SAM Geometry Distillation (saving intermediate to /tmp)
DISTILLED_DIR="/tmp/distilled_source"
echo ""
echo "=== Step 1.1: Running SAM Geometry Distillation on Graph-W and Graph-F ==="
python3 -u scripts/offline_teacher_source.py \
    --source "$SOURCE_DIR" \
    --sam-ckpt "$SAM_CKPT" \
    --sam-type vit_h \
    --out "$DISTILLED_DIR" \
    --device auto \
    --batch-size 32 \
    --mask-nms-thresh 0.50

# 6. Run Step 1 Part 2: Board-Grouped Split & 1024x1024 Tiling
echo ""
echo "=== Step 1.2: Running Board-Grouped Split (131/21) & 1024x1024 Tiling ==="
python3 -u src/semantic/prepare_source.py \
    --source "$DISTILLED_DIR" \
    --out "data/prepared_tiles" \
    --seed 42 \
    --tile 1024 \
    --overlap 256 \
    --target-val-boards 21

# 7. Package prepared tiles into downloadable archive
echo ""
echo "=== Packaging prepared tiles into tar.gz ==="
tar -czf /kaggle/working/prepared_tiles_step1.tar.gz -C data prepared_tiles
echo "Archive created: $(ls -lh /kaggle/working/prepared_tiles_step1.tar.gz 2>/dev/null || ls -lh prepared_tiles_step1.tar.gz)"

echo ""
echo "=== [Step 1] Completed Successfully! ==="
echo "Distilled source saved to: $DISTILLED_DIR (in /tmp to preserve quota)"
echo "Prepared tiles saved to:   data/prepared_tiles"
echo "Downloadable archive:      /kaggle/working/prepared_tiles_step1.tar.gz"
