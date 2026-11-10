"""Frozen checkpoint evaluation. Never tunes thresholds on the test split."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.classification.models import build_classifier
from src.classification.dataset import PCBComponentCropDataset, get_transforms
from src.classification.eval import evaluate_model
from src.classification.provenance import file_hash, new_run, write_json, digest, ROOT
from src.classification.reporting import save_records, render_records
from src.classification.policy import POLICY
from src.data_prep import validate_manifest

def evaluate_checkpoint(checkpoint, manifest, split='test', output_dir=None, batch_size=32):
    m = validate_manifest(manifest, require_images=False)
    if split == 'test' and any(not s.startswith('verified') for s in m['metadata']['official_membership'].values()):
        raise ValueError('Final test requires verified official graph membership; regenerate with source graph lists')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = ck.get('metadata', {})
    if meta.get('schema') != 'classification-checkpoint-v2' or ck.get('decision_policy') != POLICY:
        raise ValueError('Repaired checkpoint with frozen policy required; legacy diagnosis is separate')
    if meta['manifest_sha256'] != file_hash(manifest):
        raise ValueError('Frozen manifest identity mismatch')
    if meta['preprocessing'] != m['metadata']['preprocessing'] or meta['taxonomy_sha256'] != m['metadata']['taxonomy_sha256']:
        raise ValueError('Preprocessing/taxonomy mismatch')
    c = meta['effective_config']
    samples = m[split+'_crops']
    if not samples:
        raise ValueError(f'Empty eligible {split} split')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_classifier(ck['model_name'], pretrained=False, dropout_rate=c['dropout_rate']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    ds = PCBComponentCropDataset(samples, get_transforms(c['img_size'], False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, _, targets = evaluate_model(model, loader, device)
    if len(probs) != len(samples):
        raise AssertionError('Incomplete final aggregation')
    out, run_id = new_run(output_dir, 'evaluation')
    record_meta = dict(meta, evaluation_run_id=run_id, checkpoint_sha256=file_hash(checkpoint))
    save_records(out/'predictions.json', samples, probs, targets, ck['threshold'], record_meta, split)
    summary = render_records(out/'predictions.json', out/'report')
    write_json(out/'evaluation.json', summary)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--output-dir')
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    summary = evaluate_checkpoint(a.checkpoint, a.manifest, a.split, a.output_dir, a.batch_size)
    print(json.dumps({k: summary['argmax'][k] for k in ['count','accuracy','P_macro','R_macro','F1_macro','S_cls']}))

if __name__ == '__main__':
    main()
