"""Run the entire official test split only after all selections are frozen."""
import argparse
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from inference_segmentation import (read_json,save_json,load_model,predict_coco,
                                   evaluate_predictions)


def evaluate(data,out):
    import torch
    data,out=Path(data),Path(out)
    frozen=read_json(out/'selection_frozen.json')
    stats=read_json(out/'segmentation_stats.json')
    co=read_json(data/'annotations/test_boards.json')
    if len(co['images'])!=57:
        raise ValueError('Expected every official test board (47 Graph-F + 10 Graph-W)')
    for choice in frozen['trials']:
        row=next(s for s in stats if s['config_name']==choice['config_name'])
        if row['threshold']!=choice['threshold']:
            raise ValueError('Threshold changed after selection freeze')
        dest=out/row['config_name']/'test'
        if (dest/'metrics.json').exists():
            row['test_metrics']=read_json(dest/'metrics.json')
            continue
        start=time.perf_counter()
        model,meta=load_model(choice['checkpoint_name'])
        if meta['threshold']!=choice['threshold']:
            raise ValueError('Checkpoint and frozen threshold disagree')
        predictions=predict_coco(model,co,data,meta['settings'],dest,choice['threshold'])
        metrics,curve=evaluate_predictions(co,predictions,choice['threshold'])
        metrics['threshold_selected_on']='validation only'
        metrics['test_runtime_seconds']=time.perf_counter()-start
        save_json(dest/'metrics.json',metrics)
        save_json(dest/'curves.json',curve)
        row['test_metrics']=metrics
        save_json(out/'segmentation_stats.json',stats)
        del model
        torch.cuda.empty_cache()
    save_json(out/'segmentation_stats.json',stats)
    print('Test evaluation complete. Selected trial remains:',frozen['best_trial'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',default='results/segmentation')
    a=p.parse_args()
    evaluate(a.data,a.out)
