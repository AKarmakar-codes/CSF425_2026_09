"""Export exactly one manifest board's derived boxes for standalone inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import validate_manifest
from src.classification.provenance import write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--board-id', required=True)
    p.add_argument('--dataset', choices=['Graph-W','Graph-F'], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    m = validate_manifest(a.manifest, require_images=False)
    found = [b for s in ['train','val','test'] for b in m[s+'_boards'] if b['board_id'] == a.board_id and b['dataset'] == a.dataset]
    if len(found) != 1:
        raise ValueError('Expected exactly one matching board')
    write_json(a.output, [r['bbox'] for r in found[0]['annotations']])
    print(json.dumps({'image':found[0]['image_path'], 'boxes':a.output, 'count':len(found[0]['annotations'])}))

if __name__ == '__main__':
    main()
