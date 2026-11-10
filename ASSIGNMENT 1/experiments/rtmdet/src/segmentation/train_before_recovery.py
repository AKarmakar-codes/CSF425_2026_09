"""Three controlled RTMDet-Ins-tiny trials and validation-only selection."""
import argparse
import copy
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import torch
import mmdet
import mmcv
import mmengine
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import MODELS, HOOKS
from mmdet.utils import register_all_modules
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead

from inference_segmentation import (CLASSES,HEX,read_json,save_json,load_model,
    predict_coco,evaluate_predictions)

register_all_modules(init_default_scope=True)
PRETRAINED='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
TRIALS = {
    'tiny_640':dict(size=640,frozen_stages=-1,description='Baseline; all parameters trainable'),
    'tiny_960':dict(size=960,frozen_stages=-1,description='Resolution ablation; identical parameter count'),
    'tiny_960_frozen2':dict(size=960,frozen_stages=2,description='Freeze CSPNeXt stem and stages 1-2; resolution unchanged'),
}


@MODELS.register_module()
class MixedMaskRTMDetHead(RTMDetInsSepBNHead):
    """Same architecture and weights; route mask supervision by source domain."""
    def loss(self,x,batch_data_samples):
        self._mask_valid=['Graph-F__' in Path(s.metainfo['img_path']).name
                          for s in batch_data_samples]
        # Full boards can appear in explicit validation-loss calls.
        self._mask_valid=[v or '/Graph-F/' in str(s.metainfo['img_path']).replace('\\','/')
                          for v,s in zip(self._mask_valid,batch_data_samples)]
        return super().loss(x,batch_data_samples)

    def loss_mask_by_feat(self,mask_feats,flatten_kernels,sampling_results_list,
                          batch_gt_instances):
        ids=[i for i,valid in enumerate(self._mask_valid) if valid]
        if not ids:
            # Keep the mask branch in the graph while producing exactly zero loss.
            # A one-element reduction avoids FP16 overflow on large feature maps.
            return mask_feats.reshape(-1)[:1].sum()*0 + flatten_kernels.reshape(-1)[:1].sum()*0
        return super().loss_mask_by_feat(mask_feats[ids],flatten_kernels[ids],
            [sampling_results_list[i] for i in ids],[batch_gt_instances[i] for i in ids])


def number(value):
    if isinstance(value,(list,tuple)): return sum(number(v) for v in value)
    return float(value.detach().mean().cpu()) if torch.is_tensor(value) else float(value)


@HOOKS.register_module()
class AuditHistoryHook(Hook):
    """Record loss terms and real held-out validation loss without BN updates."""
    # Run before EMAHook restores raw training weights after validation.
    priority=48
    def __init__(self):
        self.train_rows=[]
        self.val_rows=[]

    def before_run(self,runner):
        path=Path(runner.work_dir)/'history.json'
        if runner._resume and path.exists():
            history=read_json(path)
            self.train_rows=history['train']
            self.val_rows=history['val']

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        if runner.iter%20==0 and outputs:
            row=dict(epoch=runner.epoch+1,iteration=runner.iter+1)
            row.update({k:number(v) for k,v in outputs.items() if 'loss' in k})
            self.train_rows.append(row)

    def after_val_epoch(self,runner,metrics=None):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        model.eval()
        sums={}
        n=0
        with torch.no_grad():
            for batch in runner.val_dataloader:
                # training=True pads GT masks, but model.eval() freezes BN statistics.
                batch=model.data_preprocessor(batch,training=True)
                with torch.autocast('cuda',enabled=torch.cuda.is_available()):
                    losses=model(**batch,mode='loss')
                batch_size=len(batch['data_samples'])
                for key,value in losses.items():
                    if 'loss' in key: sums[key]=sums.get(key,0.)+number(value)*batch_size
                n+=batch_size
        terms={k:v/max(1,n) for k,v in sums.items()}
        self.val_rows.append(dict(epoch=runner.epoch,loss=sum(terms.values()),
                                  loss_terms=terms,metrics={k:number(v) for k,v in (metrics or {}).items()}))
        save_json(Path(runner.work_dir)/'history.json',dict(train=self.train_rows,val=self.val_rows))


def make_config(data,work,trial,epochs=60,batch=2,accumulation=4,seed=42,workers=2):
    package=Path(mmdet.__file__).parent
    base=package/'.mim/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'
    if not base.exists():
        raise FileNotFoundError(f'MMDetection wheel lacks its model configs: {base}')
    cfg=Config.fromfile(str(base))
    t=TRIALS[trial]
    size=t['size']
    cfg.model.backbone.init_cfg=None
    cfg.model.backbone.frozen_stages=t['frozen_stages']
    cfg.model.backbone.norm_eval=False
    # One Colab GPU: standard BN has the same learned affine parameter count.
    cfg.model.backbone.norm_cfg=dict(type='BN',requires_grad=True)
    cfg.model.neck.norm_cfg=dict(type='BN',requires_grad=True)
    cfg.model.bbox_head.norm_cfg=dict(type='BN',requires_grad=True)
    cfg.model.bbox_head.type='MixedMaskRTMDetHead'
    cfg.model.bbox_head.num_classes=25
    cfg.model.data_preprocessor.pad_size_divisor=32
    cfg.model.data_preprocessor.pad_mask=True
    cfg.model.test_cfg.update(dict(score_thr=.001,nms_pre=3000,max_per_img=300,
                                    nms=dict(type='nms',iou_threshold=.6),mask_thr_binary=.5))
    cfg.load_from=PRETRAINED
    # No Mosaic or MixUp: they would blend valid masks with box-only labels.
    train_pipeline=[dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations',with_bbox=True,with_mask=True,poly2mask=True),
        dict(type='RandomResize',scale=(size,size),ratio_range=(.75,1.25),keep_ratio=True),
        dict(type='RandomCrop',crop_size=(size,size),recompute_bbox=True,allow_negative_crop=True),
        dict(type='FilterAnnotations',min_gt_bbox_wh=(1e-3,1e-3)),
        dict(type='YOLOXHSVRandomAug',hue_delta=3,saturation_delta=20,value_delta=20),
        dict(type='RandomFlip',prob=.75,direction=['horizontal','vertical','diagonal']),
        dict(type='Pad',size=(size,size),pad_val=dict(img=(114,114,114))),
        dict(type='PackDetInputs')]
    val_pipeline=[dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations',with_bbox=True,with_mask=True,poly2mask=True),
        dict(type='Resize',scale=(size,size),keep_ratio=True),
        dict(type='Pad',size=(size,size),pad_val=dict(img=(114,114,114))),
        dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    meta=dict(classes=CLASSES,palette=[tuple(int(h[i:i+2],16) for i in (0,2,4)) for h in HEX])
    def dataset(ann,pipeline,test_mode):
        return dict(type='CocoDataset',data_root=str(data)+os.sep,
                    ann_file='annotations/'+ann,data_prefix=dict(img=''),metainfo=meta,
                    pipeline=pipeline,test_mode=test_mode,
                    filter_cfg=None if test_mode else dict(filter_empty_gt=False,min_size=1))
    cfg.train_dataloader=dict(batch_size=batch,num_workers=workers,
        persistent_workers=workers>0,pin_memory=True,
        sampler=dict(type='DefaultSampler',shuffle=True),batch_sampler=None,
        dataset=dict(type='ClassBalancedDataset',oversample_thr=.01,
                     dataset=dataset('train_tiles.json',train_pipeline,False)))
    cfg.val_dataloader=dict(batch_size=1,num_workers=workers,persistent_workers=workers>0,
        pin_memory=True,drop_last=False,sampler=dict(type='DefaultSampler',shuffle=False),
        dataset=dataset('val_tiles_f.json',val_pipeline,True))
    cfg.test_dataloader=copy.deepcopy(cfg.val_dataloader)
    cfg.val_evaluator=dict(type='CocoMetric',ann_file=str(data/'annotations/val_tiles_f.json'),
        metric=['bbox','segm'],classwise=True,format_only=False)
    cfg.test_evaluator=copy.deepcopy(cfg.val_evaluator)
    cfg.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=epochs,val_interval=5)
    cfg.val_cfg=dict(type='ValLoop')
    cfg.test_cfg=dict(type='TestLoop')
    cfg.optim_wrapper=dict(type='AmpOptimWrapper',loss_scale='dynamic',
        accumulative_counts=accumulation,
        optimizer=dict(type='AdamW',lr=.0002,weight_decay=.05),
        paramwise_cfg=dict(norm_decay_mult=0.,bias_decay_mult=0.,bypass_duplicate=True),
        clip_grad=dict(max_norm=10,norm_type=2))
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=.1,by_epoch=True,
                             begin=0,end=min(3,epochs/2),convert_to_iter_based=True),
                         dict(type='CosineAnnealingLR',eta_min=.000002,by_epoch=True,
                              begin=min(3,epochs/2),end=epochs,convert_to_iter_based=True)]
    cfg.auto_scale_lr=dict(enable=False,base_batch_size=8)
    cfg.custom_hooks=[dict(type='EMAHook',ema_type='ExpMomentumEMA',momentum=.0002,
                           update_buffers=True,priority=49),dict(type='AuditHistoryHook')]
    cfg.default_hooks.checkpoint=dict(type='CheckpointHook',interval=5,
        max_keep_ckpts=2,save_best='coco/segm_mAP',rule='greater',save_last=True)
    cfg.default_hooks.logger=dict(type='LoggerHook',interval=20)
    cfg.visualizer=dict(type='DetLocalVisualizer',vis_backends=[dict(type='LocalVisBackend')],name='visualizer')
    cfg.work_dir=str(work)
    cfg.randomness=dict(seed=seed,deterministic=False,diff_rank_seed=False)
    cfg.env_cfg=dict(cudnn_benchmark=False,mp_cfg=dict(mp_start_method='fork',opencv_num_threads=0),
                     dist_cfg=dict(backend='nccl'))
    cfg.launcher='none'
    return cfg


def export_checkpoint(source,dest,cfg,params,settings,threshold=0.):
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    inference_cfg=dict(model=copy.deepcopy(cfg.model),test_pipeline=copy.deepcopy(cfg.val_dataloader.dataset.pipeline),
                       default_scope='mmdet')
    inference_cfg['model']['bbox_head']['type']='RTMDetInsSepBNHead'
    inference_cfg['model']['backbone']['init_cfg']=None
    ckpt.setdefault('meta',{})['pcb']=dict(inference_config=inference_cfg,
        settings=settings,threshold=threshold,classes=CLASSES,parameters=params,
        mask_supervision='Graph-F only; Graph-W boxes train classification and box branches',
        pretrained=PRETRAINED)
    # Optimizer/scheduler/EMA-training state is unnecessary for portable inference.
    for key in ('optimizer','param_schedulers','ema_state_dict','message_hub'):
        ckpt.pop(key,None)
    Path(dest).parent.mkdir(parents=True,exist_ok=True)
    torch.save(ckpt,dest)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Colab GPU runtime. CPU training is not practical.')
    if args.epochs<5:
        raise ValueError('Use at least 5 epochs so a held-out evaluation is run')
    data=Path(args.data).resolve()
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(data/'manifest.json')
    if manifest['debug_subset']:
        raise ValueError('Refusing final training on a debug subset')
    settings=dict(tile=manifest['tile'],overlap=manifest['overlap'],mask_nms_iou=.5,
                  max_detections=5000,score_floor=.001,mask_binary_threshold=.5)
    val=read_json(data/'annotations/val_boards.json')
    stats_path=out/'segmentation_stats.json'
    stats=read_json(stats_path) if stats_path.exists() else []
    completed={s['config_name'] for s in stats}
    for trial in args.trials:
        if trial in completed:
            print('Already validated:',trial,flush=True)
            continue
        start=time.perf_counter()
        work=out/'work_dirs'/trial
        cfg=make_config(data,work,trial,args.epochs,args.batch,args.accumulation,args.seed,args.workers)
        work.mkdir(parents=True,exist_ok=True)
        cfg.dump(str(work/'resolved_config.py'))
        if args.resume and (work/'last_checkpoint').exists():
            cfg.resume=True
            cfg.load_from=None
        runner=Runner.from_cfg(cfg)
        runner.model.train()  # applies CSPNeXt's frozen-stage policy before counting
        params=dict(total=sum(p.numel() for p in runner.model.parameters()),
                    trainable=sum(p.numel() for p in runner.model.parameters() if p.requires_grad))
        print(trial,params,flush=True)
        runner.train()
        best=sorted(work.glob('best_coco_segm_mAP*.pth'),key=lambda p:p.stat().st_mtime)
        if not best:
            print(
                'Best checkpoint missing. Running validation '
                'on the restored training checkpoint.',
                flush=True
            )
            if not (work / 'last_checkpoint').exists():
                raise FileNotFoundError(
                    'No resumable training checkpoint found'
                )
            runner.val()
            best = sorted(
                work.glob('best_coco_segm_mAP*.pth'),
                key=lambda p: p.stat().st_mtime
            )
            if not best:
                raise FileNotFoundError(
                    'Validation finished but did not save a best '
                    'checkpoint. Inspect validation metrics and logs.'
                )
        checkpoint=out/'checkpoints'/f'{trial}.ckpt'
        export_checkpoint(best[-1],checkpoint,cfg,params,settings)
        history=read_json(work/'history.json')
        del runner
        torch.cuda.empty_cache()
        model,meta=load_model(checkpoint)
        predictions=predict_coco(model,val,data,settings,out/trial/'validation',0.,overlays=False)
        metrics,curve=evaluate_predictions(val,predictions)
        save_json(out/trial/'validation/metrics.json',metrics)
        save_json(out/trial/'validation/curves.json',curve)
        del model
        torch.cuda.empty_cache()
        export_checkpoint(best[-1],checkpoint,cfg,params,settings,metrics['threshold'])
        best_epoch=int(re.search(r'epoch_(\d+)',best[-1].name).group(1))
        selected_train=[r['loss'] for r in history['train'] if r['epoch']==best_epoch and 'loss' in r]
        selected_val=next((r for r in history['val'] if r['epoch']==best_epoch),None)
        train_loss=sum(selected_train)/len(selected_train) if selected_train else None
        stats.append(dict(config_name=trial,description=TRIALS[trial]['description'],
            seed=args.seed,epochs=args.epochs,total_params=params['total'],
            trainable_params=params['trainable'],train_loss=train_loss,
            val_loss=selected_val['loss'] if selected_val else None,
            loss_epoch=best_epoch,train_loss_scope='Mean of logged training iterations at selected epoch',
            val_loss_scope='Entire Graph-F validation tile set, EMA weights, selected epoch',
            val_metrics=metrics,test_metrics=None,threshold=metrics['threshold'],
            data_ratio=manifest['data_ratio'],runtime_seconds=time.perf_counter()-start,
            checkpoint_name=str(checkpoint),validation_checkpoint=str(best[-1]),
            checkpoint_selection='Highest Graph-F tile validation COCO mask mAP; trial and threshold selected on full validation boards',
            settings=settings,software=dict(python=platform.python_version(),torch=torch.__version__,
                mmcv=mmcv.__version__,mmengine=mmengine.__version__,mmdet=mmdet.__version__,
                gpu=torch.cuda.get_device_name(0))))
        save_json(stats_path,stats)
    # Persist choices before any test annotation is read by the evaluation stage.
    candidates=[s for s in stats if s['config_name'] in args.trials]
    winner=max(candidates,key=lambda s:(s['val_metrics']['S_seg'],-s['trainable_params']))
    save_json(out/'selection_frozen.json',dict(best_trial=winner['config_name'],
        selection_rule='Validation S_seg; smaller trainable parameter count breaks a tie. Cohort median is unknown.',
        trials=[dict(config_name=s['config_name'],threshold=s['threshold'],
                     checkpoint_name=s['checkpoint_name'],val_S_seg=s['val_metrics']['S_seg'],
                     trainable_params=s['trainable_params']) for s in candidates],
        test_used_for_selection=False,mask_scope='Graph-F only; full assignment mask metric unavailable'))
    import shutil
    shutil.copy2(winner['checkpoint_name'],out/'checkpoints/best_segment.ckpt')
    print('Frozen selection:',winner['config_name'],'threshold',winner['threshold'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',default='results/segmentation')
    p.add_argument('--trials',nargs='+',choices=list(TRIALS),default=list(TRIALS))
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
