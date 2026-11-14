#!/usr/bin/env bash
# ==============================================================================
# CS F425 Deep Learning - Assignment II
# Step 2 Execution Script: Offline Teacher Pipeline (YOLO26-L-seg + SAM)
# ==============================================================================
set -e

echo "=== [Step 2] Starting Offline Teacher Pipeline Setup (Rule T1-T4) ==="

# 1. Environment & Dependencies Setup
echo "Installing required dependencies..."
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install ultralytics opencv-python-headless pillow scipy matplotlib tqdm pyyaml

# 2. Check GPU Availability
python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 3. Locate Prepared Tiles from Step 1
TILES_DIR="data/prepared_tiles"

if [ ! -d "$TILES_DIR" ] || [ ! -f "$TILES_DIR/train.json" ]; then
    echo "Prepared tiles not found at $TILES_DIR. Searching for prepared_tiles_step1.tar.gz..."
    ARCHIVE_PATH=""
    for cand in "/kaggle/working/prepared_tiles_step1.tar.gz" $(find /kaggle/input -name "*prepared_tiles*.tar.gz" 2>/dev/null) "prepared_tiles_step1.tar.gz"; do
        if [ -f "$cand" ]; then
            ARCHIVE_PATH="$cand"
            break
        fi
    done

    if [ -n "$ARCHIVE_PATH" ]; then
        echo "Found Step 1 archive: $ARCHIVE_PATH. Extracting to data/..."
        mkdir -p data
        tar -xzf "$ARCHIVE_PATH" -C data/
    else
        echo "[Error] Could not locate prepared tiles or prepared_tiles_step1.tar.gz."
        echo "Please ensure Step 1 has completed or attach Step 1 notebook output as input."
        exit 1
    fi
fi
echo "Verified prepared tiles at: $TILES_DIR"

# 4. Step 2.1: Export Prepared Tiles to YOLO26-seg Dataset Format (Rule T1)
YOLO_DATA_DIR="data/yolo_dataset"
echo ""
echo "=== Step 2.1: Converting Prepared Tiles to YOLO Segmentation Format ==="
python3 -u src/teacher/export_yolo_dataset.py \
    --tiles-dir "$TILES_DIR" \
    --out-dir "$YOLO_DATA_DIR"

# 5. Step 2.2: Fine-Tune YOLO26-L-seg for 25 PCB Classes (Rule T1)
mkdir -p checkpoints
TEACHER_CKPT="checkpoints/teacher_yolo26l_seg.pt"

echo ""
echo "=== Step 2.2: Fine-Tuning Teacher YOLO26-L-seg on 25 PCB Classes ==="
python3 -u scripts/train_teacher_yolo.py \
    --data "$YOLO_DATA_DIR/dataset.yaml" \
    --model "yolov8l-seg.pt" \
    --epochs 30 \
    --batch 8 \
    --imgsz 1024 \
    --device 0 \
    --out-ckpt "$TEACHER_CKPT"

# 6. Step 2.3: Verify SAM vit_h Checkpoint (Rule T2)
SAM_CKPT="/tmp/checkpoints/sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CKPT" ] || [ $(stat -c%s "$SAM_CKPT" 2>/dev/null || stat -f%z "$SAM_CKPT") -lt 1000000000 ]; then
    if [ -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
        SAM_CKPT="checkpoints/sam_vit_h_4b8939.pth"
    else
        echo "Downloading SAM vit_h weights (2.56 GB) to /tmp..."
        mkdir -p /tmp/checkpoints
        wget -c --progress=bar:force:noscroll https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O "$SAM_CKPT"
    fi
fi
echo "SAM checkpoint ready: $SAM_CKPT"

# 7. Step 2.4: Run Teacher Inference on Sample DSLR Boards (Rules T2-T4)
DSLR_SAMPLE_DIR=""
for cand in "/kaggle/input/**/pcb_dslr*" "data/pcb_dslr_download" "data"; do
    if [ -d "$cand" ]; then
        DSLR_SAMPLE_DIR="$cand"
        break
    fi
done

if [ -n "$DSLR_SAMPLE_DIR" ]; then
    echo ""
    echo "=== Step 2.4: Running Frozen Teacher on DSLR Sample Boards (Rules T2-T4) ==="
    python3 -u scripts/offline_teacher_dslr.py \
        --image "$DSLR_SAMPLE_DIR" \
        --yolo-ckpt "$TEACHER_CKPT" \
        --sam-ckpt "$SAM_CKPT" \
        --out "results/teacher_pseudo_labels" \
        --conf 0.25 \
        --mask-nms 0.50 \
        --device 0
fi

echo ""
echo "=== [Step 2] Teacher Pipeline Setup Completed Successfully! ==="
echo "Teacher YOLO26-L Checkpoint: $TEACHER_CKPT"
echo "Teacher Pseudo-Labels Saved: results/teacher_pseudo_labels"
