"""Content identities and exclusive output creation for repaired runs."""
import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def new_run(path=None, prefix='run'):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:10]
    directory = Path(path) if path else ROOT / 'runs' / (prefix + '-' + run_id)
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve(), run_id

def source_identity():
    paths = sorted(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py')) +
                   [ROOT / 'train_experiments.py', ROOT / 'inference_classification.py', ROOT/'task1results.ipynb',
                    ROOT/'scripts/run_classification.sh'])
    files = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    return {'sha256': digest(files), 'files': files, 'python': platform.python_version(),
            'git_commit': None, 'git_status': 'Archive has no Git history; content hashes identify source'}
