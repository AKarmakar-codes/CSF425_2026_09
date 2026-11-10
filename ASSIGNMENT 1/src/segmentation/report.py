"""Generate required plots and an honest machine-readable completion audit."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from inference_segmentation import read_json,save_json,CLASSES,HEX


def report(data,out):
    data,out=Path(data),Path(out)
    stats=read_json(out/'segmentation_stats.json')
    for row in stats:
        name=row['config_name']
        dest=out/name
        history=read_json(out/'work_dirs'/name/'history.json')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        train=history['train']
        for key in ('loss','loss_cls','loss_bbox','loss_mask','loss_dfl'):
            points=[p for p in train if key in p]
            if points: axes[0].plot([p['iteration'] for p in points],
                                    [p[key] for p in points],label=key,alpha=.8)
        axes[0].set(xlabel='Training iteration',ylabel='Loss',title=name)
        axes[0].legend()
        val=history['val']
        if val:
            axes[1].plot([p['epoch'] for p in val],[p['loss'] for p in val],'o-')
        axes[1].set(xlabel='Epoch',ylabel='Held-out Graph-F loss',title='Validation loss (EMA weights)')
        fig.tight_layout();fig.savefig(dest/'losses.png',dpi=160);plt.close(fig)
        for split in ('validation','test'):
            if not (dest/split/'metrics.json').exists(): continue
            metrics=read_json(dest/split/'metrics.json')
            curve=read_json(dest/split/'curves.json')
            fig,axes=plt.subplots(1,3,figsize=(15,4))
            for key,label in [('precision','Precision'),('recall','Recall'),('f1','Instance F1')]:
                axes[0].plot(curve['thresholds'],curve[key],label=label)
            axes[0].axvline(row['threshold'],color='black',linestyle='--',label='Validation t*')
            axes[0].set(xlabel='Confidence threshold',ylabel='Score',ylim=(0,1),title='Mask IoU >= 0.50')
            axes[0].legend()
            ap=metrics['standard_COCO']
            axes[1].plot(ap['recall_grid'],ap['precision_at_iou50'])
            axes[1].set(xlabel='Recall',ylabel='COCO interpolated precision',ylim=(0,1),
                        title=f"AP50={metrics['AP50']:.3f}; mAP={metrics['mAP_50_95']:.3f}")
            axes[2].plot(curve['thresholds'],curve['macro_f1'],label='Macro 25-class F1')
            axes[2].plot(curve['thresholds'],curve['f1'],label='Micro instance F1')
            axes[2].set(xlabel='Confidence threshold',ylabel='F1',ylim=(0,1))
            axes[2].legend()
            fig.suptitle(name+' / '+split+' / Graph-F reconstructed instances')
            fig.tight_layout();fig.savefig(dest/split/'PR_F1.png',dpi=160);plt.close(fig)
            colors=['#'+h for h in HEX]
            fig,axes=plt.subplots(2,1,figsize=(13,7))
            aps=[p['AP'] if p['AP'] is not None else 0 for p in metrics['class_AP']]
            f1s=[p['F1'] for p in metrics['class_F1']]
            for ax,values,label in zip(axes,(aps,f1s),('Mask AP50:95 (absent=N/A)','Mask F1 at frozen t*')):
                ax.bar(CLASSES,values,color=colors)
                ax.set(ylabel=label,ylim=(0,1))
            for i,p in enumerate(metrics['class_AP']):
                if p['AP'] is None: axes[0].text(i,.01,'N/A',rotation=90,ha='center',fontsize=7)
            fig.suptitle(name+' / '+split+' / Graph-F only')
            fig.tight_layout();fig.savefig(dest/split/'classwise_AP_F1.png',dpi=160);plt.close(fig)
    names=[r['config_name'] for r in stats]
    x=np.arange(len(stats))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(x,[r['trainable_params']/1e6 for r in stats])
    axes[0].set(xticks=x,xticklabels=names,ylabel='Trainable parameters (millions)')
    axes[1].bar(x,[r['val_metrics']['S_seg'] for r in stats],label='Validation')
    axes[1].set(xticks=x,xticklabels=names,ylabel='Validation S_seg (Graph-F only)',ylim=(0,1))
    for ax in axes: ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(out/'trial_comparison.png',dpi=160);plt.close(fig)
    audit=dict(scope='Segmentation exercise only',classification_implemented=False,
        required_trials_complete=len(stats)>=3,
        all_test_trials_complete=all(r['test_metrics'] is not None for r in stats),
        graph_w_true_pixel_masks_available=False,
        graph_f_original_instance_ids_available=False,
        full_assignment_segmentation_score_available=False,
        missing=['Graph-W true pixel-mask ground truth',
                 'Original Graph-F instance IDs/polygons to resolve touching objects',
                 'Exercise 1 classification model, metrics, plots, checkpoint and inference',
                 'Group ID, private GitHub repo, final combined assignment ZIP and email submission'],
        grading='Cohort median parameters and competing scores are unknown; no marks or rank are guaranteed.',
        no_extra_pcb_images=True,stat_file='segmentation_stats.json')
    save_json(out/'completion_audit.json',audit)
    print('Plots written. See completion_audit.json for unresolved assignment requirements.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',default='results/segmentation')
    a=p.parse_args()
    report(a.data,a.out)
