"""Render the supplied metrics and histories without running or selecting models."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODES = 'R C U Q J L RA D RN TP IC P CR M BTN FB CRA SW T F V LED S QA JP'.split()
COLORS = ['#'+c for c in 'E53935 1E88E5 00ACC1 8E24AA FB8C00 43A047 6D4C41 D81B60 3949AB C0CA33 00897B FFB300 5E35B1 039BE5 757575 FDD835 212121 7E57C2 F4511E 2E7D32 283593 7CB342 546E7A C2185B F9A825'.split()]

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)

def run(root):
    rows = read(root/'segmentation_stats.json')
    directory = root/'results/segmentation'
    for row in rows:
        name = row['config_name']
        if name == 'tiny_640':
            continue
        history = read(directory/'work_dirs'/name/'history.json')
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for key in ['loss', 'loss_cls', 'loss_bbox', 'loss_mask', 'loss_dfl']:
            points = [p for p in history['train'] if key in p]
            if points:
                axes[0].plot([p['iteration'] for p in points], [p[key] for p in points], label=key, lw=.7, alpha=.7)
        val = history['val']
        axes[0].set(xlabel='Training iteration', ylabel='Loss', title=name + ': logged losses')
        axes[0].legend(fontsize=8)
        axes[1].plot([p['epoch'] for p in val], [p['loss'] for p in val], 'o-', label='Validation loss')
        axes[1].axvline(row['loss_epoch'], color='black', ls='--', label='Selected checkpoint epoch')
        axes[1].set(xlabel='Epoch', ylabel='Loss', title='Graph-F validation, EMA weights')
        axes[1].legend(fontsize=8)
        save(fig, directory/(name+'_losses.png'))
        for split in ['validation', 'fast_test']:
            base = directory/name/split
            metric, curve = read(base/'metrics.json'), read(base/'curves.json')
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for key in ['precision', 'recall', 'f1']:
                axes[0].plot(curve['thresholds'], curve[key], label=key)
            axes[0].axvline(row['threshold'], color='black', ls='--', label='Validation t*')
            axes[0].set(xlabel='Confidence threshold', ylabel='Score', ylim=(0, 1), title='Mask IoU >= 0.50')
            axes[0].legend(fontsize=8)
            coco = metric['standard_COCO']
            axes[1].plot(coco['recall_grid'], coco['precision_at_iou50'])
            axes[1].set(xlabel='Recall', ylabel='COCO interpolated precision', ylim=(0, 1), title=f"AP50 {metric['AP50']:.4f}; mAP {metric['mAP_50_95']:.4f}")
            axes[2].plot(curve['thresholds'], curve['f1'], label='Micro instance F1')
            axes[2].plot(curve['thresholds'], curve['macro_f1'], label='Fixed-25 macro F1')
            axes[2].axvline(row['threshold'], color='black', ls='--')
            axes[2].set(xlabel='Confidence threshold', ylabel='F1', ylim=(0, 1))
            axes[2].legend(fontsize=8)
            fig.suptitle(name+' / '+split+' / Graph-F reconstructed instances')
            save(fig, base/'PR_F1.png')
            fig, axes = plt.subplots(2, 1, figsize=(13, 7))
            aps = metric['class_AP']
            f1s = metric['class_F1']
            axes[0].bar(CODES, [p['AP'] if p['AP'] is not None else 0 for p in aps], color=COLORS)
            axes[1].bar(CODES, [p['F1'] for p in f1s], color=COLORS)
            axes[0].set(ylabel='Mask AP50:95', ylim=(0, 1))
            axes[1].set(ylabel='F1 at validation-frozen t*', ylim=(0, 1))
            for i, p in enumerate(aps):
                if p['AP'] is None:
                    axes[0].text(i, .02, 'N/A', ha='center', rotation=90, fontsize=7)
            fig.suptitle(name+' / '+split+' / absent AP classes shown as N/A')
            save(fig, base/'classwise_AP_F1.png')
            with (base/'classwise_AP_F1.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['class_id', 'code', 'AP50', 'mAP50:95', 'F1', 'support'])
                for a, f in zip(aps, f1s):
                    writer.writerow([a['id'], a['code'], a['AP50'], a['AP'], f['F1'], f['support']])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = [r['config_name'].replace('yolov8m_', 'm-') for r in rows]
    x = np.arange(len(rows))
    for shift, split, color in [(-.18, 'val_metrics', '#245B85'), (.18, 'test_metrics', '#D48A38')]:
        axes[0].bar(x+shift, [r[split]['S_seg'] for r in rows], .36, color=color, label=split)
    axes[0].set(xticks=x, xticklabels=names, ylabel='Reported S_seg', ylim=(0, 1), title='Checkpoint-backed trials')
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [r['trainable_params']/1e6 for r in rows], color='#527C54')
    axes[1].set(ylabel='Trainable parameters (millions)', title='Reported training configuration')
    for shift, key, label in [(-.22, 'AP50', 'AP50'), (0, 'mAP_50_95', 'mAP50:95'), (.22, 'F1_mask', 'Mask F1')]:
        axes[2].bar(x+shift, [r['test_metrics'][key] for r in rows], .22, label=label)
    axes[2].set(xticks=x, xticklabels=names, ylabel='Saved test score', ylim=(0, 1), title='Graph-F reconstructed instances')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis='x', labelrotation=15)
    fig.suptitle('Saved results; validation selects models. Test values are descriptive only.')
    save(fig, directory/'trial_comparison.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)
