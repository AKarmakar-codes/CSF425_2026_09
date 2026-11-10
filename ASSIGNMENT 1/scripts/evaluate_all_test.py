"""Explicit classification checkpoints, frozen manifest, no segmentation side effects."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.provenance import new_run, write_json
from src.classification.evaluate import evaluate_checkpoint

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir')
    p.add_argument('--split', choices=['val','test'], default='test')
    a = p.parse_args()
    for ck in a.checkpoints:
        if not Path(ck).is_file():
            raise FileNotFoundError(ck)
    out, rid = new_run(a.output_dir, 'evaluations')
    summaries = [evaluate_checkpoint(ck, a.manifest, a.split, out/str(i)) for i, ck in enumerate(a.checkpoints)]
    write_json(out/'summary.json', {'run_id':rid, 'evaluations':summaries})
    print(out)

if __name__ == '__main__':
    main()
