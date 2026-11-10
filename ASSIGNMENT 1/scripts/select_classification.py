"""Select among completed runs using validation only and verify frozen threshold."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classification.provenance import new_run, write_json, file_hash
from src.classification.reporting import summarize_records
from src.classification.policy import sweep_threshold, predictions, POLICY
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='Individual trial directories')
    p.add_argument('--output-dir')
    a=p.parse_args()
    candidates=[]
    manifest_ids=set()
    for directory in a.runs:
        d=Path(directory)
        stats=json.loads((d/'classification_stats.json').read_text())[0]
        record=json.loads((d/'val_predictions.json').read_text())
        if record['split'] != 'val':
            raise ValueError('Selection accepts validation records only')
        ck=d/'selected.ckpt'
        if file_hash(ck) != stats['checkpoint_sha256'] or record['metadata']['checkpoint_sha256'] != stats['checkpoint_sha256']:
            raise ValueError('Checkpoint identity mismatch')
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        probs=np.array([r['probabilities'] for r in record['records']])
        targets=np.array([r['target'] for r in record['records']])
        t=sweep_threshold(probs,targets)[-1]
        if saved['threshold'] != t or saved['decision_policy'] != POLICY:
            raise ValueError('Frozen threshold/policy does not reproduce validation sweep')
        m=summarize_records(d/'val_predictions.json')['argmax']
        manifest_ids.add(saved['metadata']['manifest_sha256'])
        candidates.append({'trial':stats['config_name'],'checkpoint':str(ck.resolve()),
                           'checkpoint_sha256':file_hash(ck),'threshold':t,'validation_S_cls':m['S_cls']})
    if len(manifest_ids) != 1:
        raise ValueError('Cannot compare runs on different frozen manifests')
    best=max(candidates,key=lambda c:c['validation_S_cls'])
    out,run_id=new_run(a.output_dir,'selection')
    target=out/'best_classification.ckpt'
    shutil.copyfile(best['checkpoint'],target)
    round_trip=torch.load(target,map_location='cpu',weights_only=False)
    assert round_trip['threshold'] == best['threshold'] and file_hash(target) == best['checkpoint_sha256']
    write_json(out/'selection.json',{'run_id':run_id,'criterion':'validation argmax S_cls; input order breaks ties',
               'candidates':candidates,'selected':best,'checkpoint':str(target), 'test_used':False,
               'efficiency_median':None,'threshold_round_trip_verified':True})
    print(target)

if __name__=='__main__': main()
