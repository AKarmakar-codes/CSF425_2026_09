"""Config-driven training/validation. Test evaluation is a separate command."""
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.classification.dataset import PCBComponentCropDataset, get_transforms, compute_class_weights, create_weighted_sampler
from src.classification.models import build_classifier, count_parameters
from src.classification.loss import FocalLoss
from src.classification.eval import evaluate_model
from src.classification.policy import compute_metrics_dict, sweep_threshold, predictions, POLICY
from src.classification.preprocessing import contract
from src.classification.provenance import digest, file_hash, new_run, write_json, source_identity, ROOT
from src.classification.reporting import save_records, render_records
from src.data_prep import validate_manifest

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def effective_trial(config_path, trial, epochs=None, pretrained=None):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf8'))
    c = dict(config['classification']['trials'][trial])
    c.update(seed=config['random_seed'], trial=trial)
    c.setdefault('imbalance', 'combined')
    c.setdefault('num_workers', 0)
    c.setdefault('dropout_rate', .3)
    if epochs is not None:
        c['epochs'] = epochs
    if pretrained is not None:
        c['pretrained'] = pretrained
    if c['imbalance'] not in ['sampler_only','weighted_loss_only','combined','none']:
        raise ValueError('Unknown imbalance strategy')
    if c['loss'] not in ['focal','weighted_ce'] or c['scheduler'] != 'cosine':
        raise ValueError('Unsupported loss/scheduler')
    if c['epochs'] < 1 or c['batch_size'] < 2 or c['learning_rate'] <= 0 or c['weight_decay'] < 0:
        raise ValueError('Invalid trial configuration')
    contract(c['img_size'])
    return c

def train_one_epoch(model, loader, optimizer, criterion, device, desc='Training'):
    model.train()
    total, count = 0., 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
        count += len(images)
    return total / count

def run_classification_trial(config_name, *, manifest_path, config_path='config.yml', run_dir=None,
                             epochs=None, pretrained=None, resume=None):
    start = time.time()
    c = effective_trial(config_path, config_name, epochs, pretrained)
    set_seed(c['seed'])
    # Validate all annotations structurally, but only load train/val images here.
    m = validate_manifest(manifest_path, require_images=False)
    if contract(c['img_size']) != m['metadata']['preprocessing']:
        raise ValueError('Manifest preprocessing differs from trial; regenerate manifest')
    out, run_id = new_run(run_dir, config_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapping = json.loads((ROOT/'data_mapping.json').read_text(encoding='utf8'))
    metadata = {'schema': 'classification-checkpoint-v2', 'run_id': run_id,
                'manifest_sha256': file_hash(manifest_path), 'taxonomy_sha256': digest(mapping['classes']),
                'preprocessing': contract(c['img_size']), 'preprocessing_sha256': digest(contract(c['img_size'])),
                'effective_config': c, 'effective_config_sha256': digest(c), 'source': source_identity(),
                'class_names': [v['name'] for v in mapping['classes']], 'seed': c['seed'],
                'versions': {'torch': torch.__version__, 'numpy': np.__version__},
                'device': str(device), 'manifest_metadata': m['metadata'],
                'pretrained_weights': 'torchvision DEFAULT' if c['pretrained'] else None}
    import torchvision
    metadata['versions']['torchvision'] = torchvision.__version__
    import importlib.metadata
    for package in ['Pillow','scikit-learn','scipy','PyYAML','matplotlib','seaborn']:
        metadata['versions'][package] = importlib.metadata.version(package)
    metadata['training_augmentation'] = ['square resize','horizontal flip p=.5','vertical flip p=.5',
                                         'rotation +/-15 degrees','brightness/contrast/saturation=.2 hue=.05']
    write_json(out/'effective_config.json', metadata)
    train_samples, val_samples = m['train_crops'], m['val_crops']
    if not train_samples or not val_samples:
        raise ValueError('Train and validation splits must be nonempty')
    # Fail before training if data/cache identities are unavailable.
    from src.classification.dataset import load_sample_image
    for s in train_samples + val_samples:
        load_sample_image(s)
    def loader(samples, train=False):
        ds = PCBComponentCropDataset(samples, get_transforms(c['img_size'], train))
        sampler = create_weighted_sampler(samples) if train and c['imbalance'] in ['combined','sampler_only'] else None
        return DataLoader(ds, batch_size=c['batch_size'], sampler=sampler, shuffle=train and sampler is None,
                          num_workers=c['num_workers'])
    tr, va, tr_eval = loader(train_samples, True), loader(val_samples), loader(train_samples)
    model = build_classifier(c['backbone'], pretrained=c['pretrained'], dropout_rate=c['dropout_rate']).to(device)
    total, trainable = count_parameters(model)
    weights = compute_class_weights(train_samples).to(device) if c['imbalance'] in ['combined','weighted_loss_only'] else None
    criterion = FocalLoss(weights, c['focal_gamma']) if c['loss'] == 'focal' else torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    warmup = min(c['warmup_epochs'], c['epochs'])
    def lr_factor(index):
        if index < warmup:
            return (index + 1) / max(1, warmup)
        return .5 * (1 + math.cos(math.pi * (index - warmup) / max(1, c['epochs'] - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history, best, first_epoch, best_state = [], None, 1, None
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state.get('metadata', {}).get('schema') != metadata['schema']:
            raise ValueError('Legacy optimizer state cannot resume repaired training')
        for k in ['manifest_sha256','effective_config_sha256','taxonomy_sha256','preprocessing_sha256']:
            if state['metadata'][k] != metadata[k]:
                raise ValueError(f'Resume identity mismatch: {k}')
        if state['metadata']['source']['sha256'] != metadata['source']['sha256']:
            raise ValueError('Resume source version mismatch')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        history, best, best_state = state['history'], state['best'], state['best_model_state_dict']
        first_epoch = state['epoch'] + 1
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        metadata['resumed_checkpoint_sha256'] = file_hash(resume)
    for epoch in range(first_epoch, c['epochs'] + 1):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, tr, optimizer, criterion, device)
        vp, vi, vy, vl = evaluate_model(model, va, device, criterion)
        tp, ti, ty, tl = evaluate_model(model, tr_eval, device, criterion)
        vm, tm = compute_metrics_dict(vy, vi), compute_metrics_dict(ty, ti)
        row = dict(epoch=epoch, learning_rate=lr, train_loss=train_loss, train_eval_loss=tl,
                   val_loss=vl, train_metrics=tm, val_metrics=vm)
        history.append(row)
        if best is None or vm['F1_macro'] > best['val_metrics']['F1_macro']:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        state = {'metadata': metadata, 'model_name': c['backbone'], 'epoch': epoch,
                 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 'history': history, 'best': best,
                 'best_model_state_dict': best_state, 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}}
        torch.save(state, out/f'epoch_{epoch:04d}.resume.ckpt')
        print(f'{config_name} epoch={epoch} train_loss={train_loss:.6f} val_F1={vm["F1_macro"]:.6f}', flush=True)
    if not history:
        raise ValueError('No epochs completed')
    model.load_state_dict(best_state)
    vp, _, vy = evaluate_model(model, va, device)
    threshold = sweep_threshold(vp, vy)[-1]
    selected = {'metadata': metadata, 'model_name': c['backbone'], 'model_state_dict': best_state,
                'threshold': threshold, 'decision_policy': POLICY, 'best_epoch': best['epoch'],
                'best': best, 'selection_criterion': 'validation argmax fixed-25 macro F1; earliest epoch tie'}
    checkpoint = out/'selected.ckpt'
    torch.save(selected, checkpoint)
    reloaded = torch.load(checkpoint, map_location=device, weights_only=False)
    restored = build_classifier(reloaded['model_name'], pretrained=False, dropout_rate=c['dropout_rate']).to(device)
    restored.load_state_dict(reloaded['model_state_dict'])
    rp, _, ry = evaluate_model(restored, va, device)
    if not np.allclose(vp, rp, atol=1e-6) or not np.array_equal(predictions(vp, threshold)[1], predictions(rp, reloaded['threshold'])[1]):
        raise AssertionError('Checkpoint round-trip prediction mismatch')
    record_meta = dict(metadata, checkpoint_sha256=file_hash(checkpoint))
    save_records(out/'val_predictions.json', val_samples, rp, ry, threshold, record_meta, 'val')
    render_records(out/'val_predictions.json', out/'validation_report')
    stats = {'config_name': config_name, 'run_id': run_id, 'seed': c['seed'], 'epochs': c['epochs'],
             'total_params': total, 'trainable_params': trainable, 'best_epoch': best['epoch'], 'best': best,
             'final_epoch': history[-1], 'train_loss': best['train_loss'], 'val_loss': best['val_loss'],
             'val_metrics': best['val_metrics'], 'test_metrics': None, 'threshold': threshold, 'data_ratio': 1.,
             'runtime': (time.time()-start)/3600, 'checkpoint_name': checkpoint.name,
             'checkpoint_sha256': file_hash(checkpoint), 'selection_criterion': selected['selection_criterion'],
             'parameter_efficiency': {'median_params': None, 'status': 'cross-submission median unknown; no grading utility claimed'},
             'history': history, 'metadata': metadata, 'round_trip_verified': True}
    write_json(out/'classification_stats.json', [stats])
    import matplotlib.pyplot as plt
    for k in ['train_loss','val_loss']:
        plt.plot([h['epoch'] for h in history], [h[k] for h in history], label=k)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(out/'loss.png'); plt.close()
    for k in ['train_metrics','val_metrics']:
        plt.plot([h['epoch'] for h in history], [h[k]['accuracy'] for h in history], label=k)
    plt.xlabel('Epoch'); plt.ylabel('Argmax accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(out/'accuracy.png'); plt.close()
    return stats
