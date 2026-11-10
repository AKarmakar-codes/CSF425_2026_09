"""Audit artifacts honestly. Missing items produce a nonzero exit code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {'config_name','seed','epochs','total_params','trainable_params','train_loss','val_loss','val_metrics','test_metrics','threshold','data_ratio','runtime','checkpoint_name'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def audit(root, load_checkpoints=False):
    failures, notes, review = [], [], []
    for name in ['README.md','config.yml','data_mapping.json','classification_stats.json','segmentation_stats.json','inference_classification.py','inference_segmentation.py','run_classification.sh','run_segmentation.sh','scripts/run_classification.sh','scripts/run_segmentation.sh','checkpoints/best_classification.ckpt','checkpoints/best_segment.ckpt']:
        if not (root/name).is_file() or (root/name).stat().st_size == 0:
            failures.append('Missing required file: '+name)
    for directory in ['src/classification','src/segmentation','results/classification','results/segmentation']:
        if not (root/directory).is_dir():
            failures.append('Missing directory: '+directory)
    for name in ['classification','segmentation']:
        path=root/(name+'_stats.json')
        if not path.is_file():
            continue
        rows=json.loads(path.read_text(encoding='utf-8'))
        if len(rows)<3:
            failures.append(name+': fewer than three trials')
        for row in rows:
            missing=REQUIRED_KEYS-row.keys()
            if missing:
                failures.append(name+'/'+row.get('config_name','?')+': missing JSON keys '+str(sorted(missing)))
            if name=='classification':
                ck=root/row['checkpoint_name']
                if not ck.is_file():
                    failures.append('Missing classification trial checkpoint: '+row['checkpoint_name'])
                elif row.get('checkpoint_sha256') and sha(ck)!=row['checkpoint_sha256']:
                    failures.append('Classification trial checkpoint hash mismatch: '+row['config_name'])
                if not row.get('checkpoint_metadata',{}).get('report_matches_checkpoint_f1',False):
                    notes.append(row['config_name']+': supplied checkpoint validation F1 differs from historical quantitative report; see checkpoint_metadata')
                    review.append('Reconcile '+row['config_name']+' historical metrics with the supplied checkpoint before claiming its old score')
        if not any((root/'results'/name).glob('*.png')):
            failures.append(name+': no top-level result PNG')
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower()=='.json':
            try:
                json.loads(p.read_text(encoding='utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value: '+value)))
            except Exception as e:
                failures.append(str(p.relative_to(root))+': invalid JSON '+str(e))
        if p.suffix=='.py':
            try:
                ast.parse(p.read_text(encoding='utf-8-sig'))
            except SyntaxError as e:
                failures.append(str(p.relative_to(root))+': syntax error '+str(e))
        if p.suffix=='.sh' and (b'\r\n' in p.read_bytes() or p.read_bytes().startswith(b'\xef\xbb\xbf')):
            failures.append(str(p.relative_to(root))+': nonportable shell encoding')
        if p.suffix.lower() in {'.pem','.key','.pcb64','.zip'} or p.name.startswith('csisdl_') or p.name in {'id_rsa','id_ed25519','.env'}:
            failures.append('Forbidden file: '+str(p.relative_to(root)))
        if '.git' in p.relative_to(root).parts:
            failures.append('Development file: '+str(p.relative_to(root)))
    for name in ['inference_classification.py','inference_segmentation.py']:
        modules=[]
        for node in ast.walk(ast.parse((root/name).read_text())):
            if isinstance(node,ast.Import):
                modules += [n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                modules.append(node.module or '')
        if any(m.split('.')[0] in {'src','inference_classification','inference_segmentation','persistence'} for m in modules):
            failures.append(name+': imports a project-specific module')
    selection=json.loads((root/'selection_segmentation.json').read_text())
    checkpoint=root/'checkpoints/best_segment.ckpt'
    if checkpoint.exists() and sha(checkpoint)!=selection['best_checkpoint_sha256']:
        failures.append('Best segmentation checkpoint hash mismatch')
    classification_selection=json.loads((root/'selection_classification.json').read_text())
    checkpoint=root/'checkpoints/best_classification.ckpt'
    if checkpoint.is_file() and sha(checkpoint)!=classification_selection.get('best_checkpoint_sha256'):
        failures.append('Best classification checkpoint hash mismatch')
    metadata=json.loads((root/'submission_metadata.json').read_text())
    if not metadata.get('final_private_repository_url'):
        notes.append('Final private CSF425_2026_G09 repository URL not yet supplied')
        review.append('Create or verify the final private repository separately')
    if load_checkpoints:
        import torch
        for name in ['best_classification.ckpt','best_segment.ckpt']:
            p=root/'checkpoints'/name
            if p.exists():
                try:
                    d=torch.load(p,map_location='cpu',weights_only=False)
                    if not isinstance(d,dict):
                        failures.append(name+': expected checkpoint mapping')
                    if name=='best_classification.ckpt':
                        sys.path.insert(0,str(root))
                        from inference_classification import validate_checkpoint_contract
                        validate_checkpoint_contract(d)
                        if d['threshold']!=classification_selection['frozen_threshold']:
                            failures.append('Best classification frozen threshold differs from selection record')
                except Exception as e:
                    failures.append(name+': unreadable: '+str(e))
    notes += [
        'Classification trained weights loaded and tested; historical ConvNeXt metrics/plots have a checkpoint F1/threshold mismatch and were not silently relabeled as reevaluated.',
        'Supplied YOLO-s weights are from a 25-epoch run; older 70-epoch s and nano reports are not primary verified results.',
        'Graph-W true pixel mask ground truth is absent; Graph-F instances are reconstructed from class masks.',
        'A clean dependency installation and the separate MMDetection inference environment were not verified during packaging.'
    ]
    return {'artifact_checks_pass':not failures, 'ready_to_zip':not failures,
            'ready_to_submit':not failures and not review, 'submission_review_required':review,
            'required_file_failures':failures, 'verification_limits':notes,
            'checkpoint_loading_requested':load_checkpoints,'file_count':sum(p.is_file() for p in root.rglob('*'))}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--load-checkpoints',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    report=audit(a.root,a.load_checkpoints)
    text=json.dumps(report,indent=2)
    print(text)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text+'\n',encoding='utf-8')
    sys.exit(0 if report['artifact_checks_pass'] else 1)
