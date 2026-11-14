#!/usr/bin/env bash
# ==============================================================================
# Package Assignment II Code for Kaggle Upload
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

ZIP_NAME="CSF425_Assignment2_Code.zip"
echo "Creating lightweight upload bundle: $ZIP_NAME..."

python3 - << 'EOF'
import os, zipfile
from pathlib import Path

zip_path = Path("CSF425_Assignment2_Code.zip")
if zip_path.exists():
    zip_path.unlink()

include_paths = [
    Path("src"),
    Path("scripts"),
    Path("data_mapping.json"),
    Path("config.yml"),
    Path("requirements.txt"),
    Path("plan.txt"),
    Path("step1_kaggle_notebook.ipynb")
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in include_paths:
        if item.is_file():
            zf.write(item, item.name)
        elif item.is_dir():
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                for f in files:
                    if f.endswith(".pyc") or f.startswith("."):
                        continue
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(item.parent)
                    zf.write(full_p, rel_p.as_posix())

print(f"Successfully packaged {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
EOF

echo "Done! You can upload CSF425_Assignment2_Code.zip directly to Kaggle in seconds."
