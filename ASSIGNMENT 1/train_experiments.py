"""Classification trains/validates only. Final test is explicitly separate."""
import argparse
import json
from pathlib import Path
from src.classification.provenance import new_run, write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['classification'], default='classification')
    p.add_argument('--config', default='config.yml')
    p.add_argument('--manifest')
    p.add_argument('--trials', nargs='+')
    p.add_argument('--epochs', type=int)
    p.add_argument('--run-dir')
    p.add_argument('--skip-test', action='store_true', help='Compatibility flag; test is always separate')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--resume', help='Repaired epoch resume checkpoint; requires one trial')
    p.add_argument('--synthetic-smoke', action='store_true')
    a = p.parse_args()
    out, run_id = new_run(a.run_dir, 'experiments')
    if a.synthetic_smoke:
        import torch
        torch.set_num_threads(2)
        from src.classification.smoke import assets
        a.manifest, a.config = assets(out/'synthetic_assets')
        a.trials = ['synthetic_smoke']
    if not a.manifest:
        p.error('--manifest is required (derive repaired classification manifest first)')
    import yaml
    config = yaml.safe_load(Path(a.config).read_text(encoding='utf8'))
    trials = a.trials or list(config['classification']['trials'])
    if a.resume and len(trials) != 1:
        p.error('--resume requires exactly one trial')
    from src.classification.train import run_classification_trial
    stats = []
    for trial in trials:
        if trial not in config['classification']['trials']:
            raise ValueError(f'Unknown trial: {trial}')
        stats.append(run_classification_trial(trial, manifest_path=a.manifest, config_path=a.config,
                     run_dir=out/trial, epochs=a.epochs, pretrained=False if a.no_pretrained else None, resume=a.resume))
    best = max(stats, key=lambda s: s['val_metrics']['S_cls'])
    write_json(out/'selection.json', {'run_id': run_id, 'criterion': 'validation argmax S_cls; configured order breaks ties',
               'efficiency_median': None, 'selected_trial': best['config_name'],
               'selected_checkpoint': str(out/best['config_name']/'selected.ckpt'),
               'checkpoint_sha256': best['checkpoint_sha256'], 'test_used_for_selection': False,
               'synthetic': a.synthetic_smoke})
    print(out)

if __name__ == '__main__':
    main()
