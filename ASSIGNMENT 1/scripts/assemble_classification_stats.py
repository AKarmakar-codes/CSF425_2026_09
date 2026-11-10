"""Assemble the assignment statistics from matched training and final test evidence."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.provenance import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-stats', nargs='+', required=True)
    p.add_argument('--evaluations', nargs='+', required=True, help='evaluation.json paths')
    p.add_argument('--output', required=True)
    a=p.parse_args()
    evaluations={}
    for path in a.evaluations:
        e=json.loads(Path(path).read_text(encoding='utf8'))
        if e['split'] != 'test':
            raise ValueError('Final packaging requires test evaluation, never validation metrics relabeled as test')
        key=e['metadata']['checkpoint_sha256']
        if key in evaluations:
            raise ValueError('Duplicate checkpoint evaluation')
        evaluations[key]=e
    rows=[]
    for path in a.training_stats:
        for s in json.loads(Path(path).read_text(encoding='utf8')):
            e=evaluations.pop(s['checkpoint_sha256'])
            if s['metadata']['manifest_sha256'] != e['metadata']['manifest_sha256'] or s['threshold'] != e['threshold']:
                raise ValueError('Training/test manifest or threshold mismatch')
            s.update(test_metrics=e['argmax'], primary_test_policy='argmax',
                     supplementary_frozen_threshold=e['frozen_threshold'],
                     test_prediction_sha256=e['prediction_sha256'], test_evaluation_metadata=e['metadata'])
            rows.append(s)
    if evaluations:
        raise ValueError('Unmatched evaluation evidence')
    write_json(a.output,rows)

if __name__=='__main__': main()
