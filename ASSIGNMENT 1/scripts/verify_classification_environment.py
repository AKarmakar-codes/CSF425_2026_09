"""Print exact dependency/device/source versions; fail on missing required packages."""
import importlib
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.provenance import source_identity

def main():
    versions = {n: str(getattr(importlib.import_module(n), '__version__', 'unknown'))
                for n in ['torch','torchvision','numpy','scipy','sklearn','PIL','yaml','matplotlib','seaborn']}
    import torch
    print(json.dumps({'python':platform.python_version(), 'executable':sys.executable, 'packages':versions,
                      'cuda':torch.cuda.is_available(), 'source':source_identity()},indent=2))

if __name__ == '__main__':
    main()
