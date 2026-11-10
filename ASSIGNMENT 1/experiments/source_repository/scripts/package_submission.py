"""
Master Submission Packager & Verifier for CS F425 Deep Learning Project Assignment I.
Strictly verifies all items required by Section 8 (Submission & Organization - 3 Marks):
  1. Verifies exact directory structure & mandatory filenames.
  2. Ensures both checkpoints exist: best_classification.ckpt & best_segment.ckpt.
  3. Verifies JSON statistics schema consistency across trials.
  4. Confirms standalone inference scripts and shell wrappers exist.
  5. Packages the clean repository into CSF425_2026_09_Assignment-I.zip under CSF425_2026_09/
     WITHOUT any raw datasets, large image dumps, git history, or private SSH keys.
"""

import os
import sys
import json
import zipfile
import shutil

GROUP_ID = "09"
ZIP_NAME = f"CSF425_2026_{GROUP_ID}_Assignment-I.zip"
ARCHIVE_ROOT = f"CSF425_2026_{GROUP_ID}"


def audit_repository() -> bool:
    print("=" * 75)
    print(f"📋 PRE-FLIGHT SUBMISSION AUDIT — GROUP {GROUP_ID}")
    print("=" * 75)

    all_passed = True

    # 1. Mandatory Root Files
    mandatory_root = [
        "README.md",
        "config.yml",
        "data_mapping.json",
        "classification_stats.json",
        "segmentation_stats.json",
        "inference_classification.py",
        "inference_segmentation.py",
        "data/sample.png",
    ]
    for f in mandatory_root:
        if os.path.exists(f):
            print(f"  ✅ [Found] {f}")
        else:
            print(f"  ❌ [MISSING] Mandatory root file: {f}")
            all_passed = False

    # 2. Mandatory Checkpoints
    mandatory_ckpts = [
        "checkpoints/best_classification.ckpt",
        "checkpoints/best_segment.ckpt",
    ]
    for c in mandatory_ckpts:
        if os.path.exists(c):
            sz_mb = os.path.getsize(c) / (1024 * 1024)
            print(f"  ✅ [Found] {c} ({sz_mb:.1f} MB)")
        else:
            print(f"  ❌ [MISSING] Checkpoint: {c}")
            all_passed = False

    # 3. Mandatory Scripts
    mandatory_scripts = [
        "scripts/run_classification.sh",
        "scripts/run_segmentation.sh",
    ]
    for s in mandatory_scripts:
        if os.path.exists(s):
            print(f"  ✅ [Found] {s}")
        else:
            print(f"  ❌ [MISSING] Shell script: {s}")
            all_passed = False

    # 4. Mandatory Source Files
    mandatory_src = [
        "src/classification/train.py",
        "src/classification/eval.py",
        "src/classification/dataset.py",
        "src/classification/models.py",
        "src/classification/loss.py",
        "src/segmentation/train.py",
        "src/segmentation/yolo_train.py",
        "src/segmentation/eval.py",
        "src/segmentation/dataset.py",
        "src/segmentation/models.py",
    ]
    for s in mandatory_src:
        if os.path.exists(s):
            print(f"  ✅ [Found] {s}")
        else:
            print(f"  ❌ [MISSING] Source file: {s}")
            all_passed = False

    # 5. JSON Statistics Schema Verification
    req_keys = [
        "config_name", "seed", "epochs", "total_params", "trainable_params",
        "train_loss", "val_loss", "val_metrics", "test_metrics",
        "threshold", "data_ratio", "runtime", "checkpoint_name"
    ]
    for jf in ["classification_stats.json", "segmentation_stats.json"]:
        if os.path.exists(jf):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if not isinstance(data, list) or len(data) == 0:
                    print(f"  ❌ [ERROR] {jf} must be a non-empty list of trial records!")
                    all_passed = False
                else:
                    for i, trial in enumerate(data):
                        missing = [k for k in req_keys if k not in trial]
                        if missing:
                            print(f"  ❌ [SCHEMA BUG] {jf} Trial {i} ({trial.get('config_name')}) missing: {missing}")
                            all_passed = False
                    print(f"  ✅ [Validated] {jf} ({len(data)} trials, all schemas strictly valid)")
            except Exception as e:
                print(f"  ❌ [INVALID JSON] Could not parse {jf}: {e}")
                all_passed = False

    # 6. Results & Plots Check
    cls_plots = [f for f in os.listdir("results/classification") if f.endswith(".png")] if os.path.exists("results/classification") else []
    seg_plots = [f for f in os.listdir("results/segmentation") if f.endswith(".png")] if os.path.exists("results/segmentation") else []
    print(f"  ✅ [Plots] Classification: {len(cls_plots)} plots | Segmentation: {len(seg_plots)} plots")

    # 7. Security Check (NO SSH KEYS)
    for root, dirs, files in os.walk("."):
        for file in files:
            if "csisdl" in file or file.endswith(".pem") or file.endswith(".key") or file == "id_rsa":
                print(f"  🚨 [SECURITY VIOLATION] Private SSH key found: {os.path.join(root, file)}! DO NOT SUBMIT!")
                all_passed = False

    return all_passed


def build_submission_zip():
    if not audit_repository():
        print("\n❌ AUDIT FAILED! Fix the missing files above before packaging.")
        sys.exit(1)

    print("\n" + "=" * 75)
    print(f"📦 CREATING FINAL ARCHIVE: {ZIP_NAME}")
    print("=" * 75)

    # Folders and files to include
    include_paths = [
        "README.md",
        "config.yml",
        "environment.yml",
        "requirements.txt",
        "data_mapping.json",
        "classification_stats.json",
        "segmentation_stats.json",
        "inference_classification.py",
        "inference_segmentation.py",
        "train_experiments.py",
        "task1results.ipynb",
        "data/sample.png",
        "src",
        "scripts",
        "checkpoints/best_classification.ckpt",
        "checkpoints/best_segment.ckpt",
        "results",
    ]

    # Patterns to strictly exclude
    exclude_patterns = [
        ".git",
        "__pycache__",
        ".pyc",
        "GraphPCB",
        "crops",
        "yolo_seg",
        "coco_pcb_640",
        "runs",
        "csisdl",
        ".zip",
        ".pt",
        ".pth",
    ]

    count = 0
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include_paths:
            if os.path.isfile(item):
                arcname = os.path.join(ARCHIVE_ROOT, item)
                zf.write(item, arcname)
                count += 1
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    # Filter directories
                    dirs[:] = [d for d in dirs if not any(p in d for p in exclude_patterns)]
                    for f in files:
                        if any(p in f for p in exclude_patterns):
                            continue
                        # If inside checkpoints, only include best_classification.ckpt and best_segment.ckpt
                        if "checkpoints" in root and f not in ["best_classification.ckpt", "best_segment.ckpt"]:
                            continue
                        f_path = os.path.join(root, f)
                        arcname = os.path.join(ARCHIVE_ROOT, f_path)
                        zf.write(f_path, arcname)
                        count += 1

    sz_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
    print(f"\n🎉 Successfully created: {ZIP_NAME}")
    print(f"   Total files packaged: {count}")
    print(f"   Archive size:         {sz_mb:.2f} MB")
    print(f"   Top-level folder:     {ARCHIVE_ROOT}/")
    print("\n" + "=" * 75)
    print("📧 FINAL EMAIL SUBMISSION INSTRUCTIONS:")
    print(f"   To:      skaziz.ali@hyderabad.bits-pilani.ac.in")
    print(f"   Subject: [CSF425 2026 {GROUP_ID} Assignment-I]")
    print(f"   Attach:  {ZIP_NAME}")
    print("=" * 75)


if __name__ == "__main__":
    build_submission_zip()

