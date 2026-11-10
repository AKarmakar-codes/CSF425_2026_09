"""Resumable YOLOv8m-seg trials using exact COCO masks and mixed supervision."""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
import os
import signal
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from ultralytics.utils.torch_utils import ModelEMA
from data import CocoTiles,collate,to_device
from model import make_model,MixedMaskLoss,set_train_mode,export
from inference_segmentation import (CLASSES,read_json,save_json,coco_metrics,
    predict_coco,evaluate_predictions,load_model)
from persistence import atomic_json,BackupManager,SessionBudget,make_store,sha256,validate_local_commits
save_json=atomic_json

TRIALS={
    'yolov8m_640_aug':dict(size=640,freeze=0),
    'yolov8m_960_aug':dict(size=960,freeze=0),
    'yolov8m_960_aug_frozen_backbone':dict(size=960,freeze=10),
}
LOSS_KEYS=('loss_bbox','loss_mask','loss_cls','loss_dfl')


class SessionStop(RuntimeError): pass


def split_fingerprint(manifest):
    rows=sorted((r['domain'],r['stem'],r['split']) for r in manifest['records'])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


class EpochSampler(Sampler):
    """Send epoch with each index so persistent worker RNG stays correct."""
    def __init__(self,dataset,seed):
        self.dataset=dataset;self.seed=seed;self.epoch=0
    def __len__(self): return len(self.dataset)
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        for index in torch.randperm(len(self.dataset),generator=generator).tolist():
            yield self.epoch,index


def loader(dataset,batch,workers,shuffle,seed):
    sampler=EpochSampler(dataset,seed) if shuffle else None
    return DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,
        pin_memory=torch.cuda.is_available(),collate_fn=collate,persistent_workers=workers>0,
        **(dict(prefetch_factor=2) if workers>0 else {}),
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed))


def val_loss(model,dl,device,stop_check=None):
    model.eval();criterion=MixedMaskLoss(model);sums=torch.zeros(4,device=device);n=0
    with torch.no_grad():
        for raw in dl:
            if stop_check: stop_check()
            batch=to_device(raw,device)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                _,terms=criterion(model(batch['img']),batch)
            b=len(batch['img']);sums+=terms.detach().float()*b;n+=b
    return dict(zip(LOSS_KEYS,(sums/max(1,n)).cpu().tolist()))


def atomic_torch_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def run(args):
    config_path=getattr(args,'backup_config',None) or os.environ.get('GRAPHPCB_BACKUP_CONFIG')
    budget=SessionBudget(args.hours,getattr(args,'shutdown_buffer_seconds',900) if args.hours else 0)
    backup=None;interrupted=[False];previous_handlers={}
    if config_path:
        config=read_json(config_path)
        if Path(args.out).resolve()!=(Path(config['run_root'])/'results/segmentation').resolve():
            raise ValueError('Backup run_root does not match trainer output directory')
        validate_local_commits(config['run_root'])
        workspace=Path(config['workspace']);store=make_store(config,workspace)
        backup=BackupManager(config['run_root'],store,workspace,
            config.get('interval_seconds',3600),config.get('retry_seconds',60)).start()
    else: print('WARNING: persistent storage is NOT configured; local checkpoints alone do not survive session loss',flush=True)
    def stop_request(signum,frame):
        interrupted[0]=True
        print('Shutdown signal received; rolling back unfinished epoch to its last completed checkpoint',flush=True)
    if __import__('threading').current_thread() is __import__('threading').main_thread():
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[sig]=signal.signal(sig,stop_request)
    try:
        return _run(args,backup,budget,interrupted)
    except SessionStop:
        save_json(Path(args.out)/'training_state.json',dict(complete=False,
            reason='Validation interrupted for shutdown reserve; resume from last committed epoch'))
        print('Stopping validation safely; final persistent backup follows',flush=True)
    finally:
        if backup:
            success=backup.finish(budget.remaining(getattr(args,'shutdown_buffer_seconds',900)))
            status=read_json(Path(args.out)/'training_state.json') if (Path(args.out)/'training_state.json').exists() else {}
            status.update(persistent_backup=dict(backup.status),final_persistent_backup_confirmed=success)
            save_json(Path(args.out)/'training_state.json',status)
            print('Final persistent backup status:',backup.status,flush=True)
        for sig,handler in previous_handlers.items(): signal.signal(sig,handler)


def _run(args,backup,budget,interrupted):
    import ultralytics
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('Enable a Kaggle GPU accelerator before training')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    data,out=Path(args.data).resolve(),Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(data/'manifest.json')
    if manifest.get('debug_subset'): raise ValueError('Debug data is not a final experiment')
    smoke=getattr(args,'persistence_smoke',False)
    if manifest.get('persistence_smoke_fixture') and not smoke: raise ValueError('Synthetic reliability fixtures are not assignment data')
    if not smoke and (args.epochs<5 or args.epochs%5): raise ValueError('Use at least 5 epochs, in multiples of 5')
    protocol=split_fingerprint(manifest)
    invocation_start=time.perf_counter()
    stats_path=out/'segmentation_stats.json'
    stats=read_json(stats_path) if stats_path.exists() else []
    for row in stats:
        checkpoint=out/'checkpoints'/f'{row["config_name"]}.ckpt'
        if not checkpoint.exists(): raise FileNotFoundError('Completed trial checkpoint is missing; restore a verified backup')
        row['checkpoint_name']=str(checkpoint)
    def stopped(trial,completed,reason):
        save_json(out/'training_state.json',dict(complete=False,current_trial=trial,
            completed_epoch=completed,reason=reason,
            recovery='Next epoch after last completed checkpoint; unfinished work is repeated'))
        print(f'Stopping safely: {trial}, completed epoch {completed}; {reason}',flush=True)
    def stop_due(): return interrupted[0] or budget.should_stop()
    def check_budget():
        if stop_due(): raise SessionStop()
    def publication(): return backup.lock if backup else nullcontext()
    for trial in args.trials:
        if any(s['config_name']==trial for s in stats):
            print('Already validated:',trial,flush=True);continue
        t=TRIALS[trial];work=out/'work_dirs'/trial;work.mkdir(parents=True,exist_ok=True)
        last=work/'last.pt';best=out/'checkpoints'/f'{trial}.ckpt'
        train_ds=CocoTiles(data,'train_tiles.json',t['size'],True,args.seed,True)
        val_ds=CocoTiles(data,'val_tiles_f.json',t['size'])
        settings=dict(tile=manifest['tile'],overlap=manifest['overlap'],mask_nms_iou=.5,
                      max_detections=5000,score_floor=.001,max_per_tile=300,
                      mask_binary_threshold=.5,input_size=t['size'])
        config=dict(architecture='yolov8m-seg',trial=trial,epochs=args.epochs,batch=args.batch,
            accumulation=args.accumulation,seed=args.seed,split_fingerprint=protocol,
            lr=.0002,weight_decay=.05,weight_decay_exclusions='bias and normalization affine (1D parameters)',
            freeze=t['freeze'],settings=settings,
            augmentation=dict(degrees=15,scale=[.75,1.25],translate=.1,fliplr=.5,flipud=.5,
                              hue_delta=3,saturation_gain=[.8,1.2],value_gain=[.8,1.2],mosaic=0,mixup=0),
            class_balance='Ceil(max(1,sqrt(0.01 / per-tile class presence frequency)))')
        config['pretrained_weights']=Path(args.weights).name if args.weights else None
        if (work/'config.json').exists() and read_json(work/'config.json')!=config:
            raise ValueError('Run settings changed; choose a new output run directory')
        save_json(work/'config.json',config)
        resumed=last.exists() and args.resume
        if last.exists() and not args.resume: raise FileExistsError('Use --resume or a new output directory')
        model=make_model(None if resumed else args.weights,t['freeze']).to(device)
        params=dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(trial,params,flush=True)
        decay=[p for p in model.parameters() if p.requires_grad and p.ndim>1]
        no_decay=[p for p in model.parameters() if p.requires_grad and p.ndim<=1]
        optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.05),
                                    dict(params=no_decay,weight_decay=0.)],lr=.0002,
                                    **(dict(fused=True) if device.type=='cuda' else {}))
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        ema=ModelEMA(model);criterion=MixedMaskLoss(model)
        history=dict(train=[],val=[]);start_epoch=0;best_ap=-1.;best_epoch=0;runtime=0.
        if resumed:
            c=torch.load(last,map_location=device,weights_only=False)
            if c['config']!=config: raise ValueError('Resume config mismatch')
            model.load_state_dict(c['model']);ema.ema.load_state_dict(c['ema'])
            ema.updates=c['ema_updates'];optimizer.load_state_dict(c['optimizer']);scaler.load_state_dict(c['scaler'])
            if device.type=='cuda':
                for group in optimizer.param_groups: group['fused']=True
            start_epoch=c['epoch'];best_ap=c['best_ap'];best_epoch=c['best_epoch']
            history=c['history'];runtime=c['runtime_seconds']
            torch.set_rng_state(c['rng_cpu'].cpu())
            if 'rng_python' in c: random.setstate(c['rng_python'])
            if 'rng_numpy' in c: np.random.set_state(c['rng_numpy'])
            if 'rng_python' not in c or 'rng_numpy' not in c:
                print('WARNING: legacy checkpoint lacks Python/NumPy RNG state; bitwise continuation of that older run is unavailable',flush=True)
            # Only GPU 0 is used; saved outputs may be resumed on a 1- or 2-GPU allocation.
            if device.type=='cuda' and c['rng_cuda']: torch.cuda.set_rng_state(c['rng_cuda'][0].cpu(),0)
            scheduler=c.get('scheduler')
            saved_sampler=c.get('sampler')
            if scheduler and (scheduler['completed_epoch']!=start_epoch or scheduler['total_epochs']!=args.epochs):
                raise ValueError('Saved scheduler position does not match experiment')
            print(f'Resuming {trial} from completed epoch {start_epoch}; next epoch {start_epoch+1}',flush=True)
            del c
        trial_start=time.perf_counter()
        val_dl=loader(val_ds,1,args.workers,False,args.seed)
        dl=loader(train_ds,args.batch,args.workers,True,args.seed)
        if resumed and scheduler and scheduler['batches_per_epoch']!=len(dl):
            raise ValueError('Restored schedule loader length mismatch')
        if resumed and saved_sampler and (saved_sampler['next_epoch']!=start_epoch or saved_sampler['seed']!=args.seed):
            raise ValueError('Restored sampler position mismatch')
        batch=loss=weighted=None
        def checkpoint(epoch,elapsed):
            # LR schedule is analytic, not a torch Scheduler object. Its state is
            # fully specified by the saved epoch, loader length and constants.
            return dict(model=model.state_dict(),ema=ema.ema.state_dict(),
                ema_updates=ema.updates,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                epoch=epoch,best_ap=best_ap,best_epoch=best_epoch,history=history,config=config,
                runtime_seconds=elapsed,rng_cpu=torch.get_rng_state(),rng_python=random.getstate(),
                rng_numpy=np.random.get_state(),rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                sampler=dict(seed=args.seed,next_epoch=epoch,batches_per_epoch=len(dl)),
                scheduler=dict(kind='analytic_warmup_cosine_v1',completed_epoch=epoch,total_epochs=args.epochs,
                    batches_per_epoch=len(dl),base_lr=.0002,min_lr=.000002,warmup_epochs=3.),
                recovery='Epoch boundary only; no unfinished gradients or batches are restored')
        def commit(epoch):
            paths=[last]+([best,work/'best.pt'] if best.exists() else [])
            save_json(work/'commit.json',dict(trial=trial,epoch=epoch,best_ap=best_ap,
                committed_utc=__import__('persistence').utc(),published_hashes={
                    p.relative_to(out.parents[1]).as_posix():sha256(p) for p in paths if p.exists()}))
        # A last-completed-epoch-0 checkpoint protects even an exceptionally long
        # first epoch. Legacy checkpoints are committed before hourly snapshots.
        with publication():
            if not last.exists(): atomic_torch_save(checkpoint(0,runtime),last)
            commit(start_epoch)
        if backup: backup.request()
        for epoch in range(start_epoch,args.epochs):
            if stop_due(): stopped(trial,epoch,'Shutdown/time reserve reached');return
            dl.sampler.epoch=epoch
            epoch_start=time.perf_counter()
            set_train_mode(model,t['freeze']);optimizer.zero_grad(set_to_none=True)
            sums=torch.zeros(4,device=device);images_seen=0
            for step,raw in enumerate(dl):
                if stop_due(): stopped(trial,epoch,'Unfinished epoch discarded to reserve final upload time');return
                batch=to_device(raw,device)
                progress=(epoch+step/max(1,len(dl)))/args.epochs
                lr=.000002+.5*(.0002-.000002)*(1+math.cos(math.pi*progress))
                lr*=min(1.,(epoch+step/max(1,len(dl))+.1)/3.)
                for group in optimizer.param_groups: group['lr']=lr
                group_len=min(args.accumulation,len(dl)-(step//args.accumulation)*args.accumulation)
                with torch.autocast(device.type,enabled=device.type=='cuda'):
                    weighted,terms=criterion(model(batch['img']),batch)
                    loss=weighted.sum()/len(batch['img'])
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {trial}/{epoch+1}/{step}')
                scaler.scale(loss/group_len).backward()
                if (step+1)%args.accumulation==0 or step+1==len(dl):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=previous: ema.update(model)
                    optimizer.zero_grad(set_to_none=True)
                b=len(batch['img']);sums+=terms.detach().float()*b;images_seen+=b
                if step%20==0:
                    row=dict(epoch=epoch+1,iteration=epoch*len(dl)+step+1,loss=float(terms.sum()))
                    row.update(dict(zip(LOSS_KEYS,terms.float().cpu().tolist())))
                    history['train'].append(row)
                    print(f'{trial} epoch {epoch+1}/{args.epochs} batch {step+1}/{len(dl)} loss={row["loss"]:.4f}',flush=True)
            if device.type=='cuda': torch.cuda.synchronize()
            train_seconds=time.perf_counter()-epoch_start
            throughput=dict(epoch=epoch+1,training_seconds=train_seconds,
                            images=images_seen,images_per_second=images_seen/max(train_seconds,1e-9))
            history.setdefault('throughput',[]).append(throughput)
            print('Training throughput:',throughput,flush=True)
            epoch_terms=dict(zip(LOSS_KEYS,(sums/max(1,images_seen)).cpu().tolist()))
            improved=False
            if (epoch+1)%5==0 or (smoke and epoch+1==args.epochs):
                if stop_due(): stopped(trial,epoch,'Before validation; repeat unfinished epoch');return
                ema.ema.eval()
                terms=val_loss(ema.ema,val_dl,device,check_budget)
                co=val_ds.co
                predictions=predict_coco(ema.ema,co,data,settings,work/'tile_validation',0.,overlays=False,stop_check=check_budget)
                metrics=coco_metrics(co,predictions)
                ap=metrics['mAP_50_95']
                history['val'].append(dict(epoch=epoch+1,loss=sum(terms.values()),loss_terms=terms,
                    train_epoch_loss_terms=epoch_terms,metrics=dict(mask_mAP=ap,mask_AP50=metrics['AP50'])))
                if ap is not None and ap>best_ap:
                    best_ap=ap;best_epoch=epoch+1
                    improved=True
                print('Validation tile mask mAP:',ap,'best epoch:',best_epoch,flush=True)
            elapsed=runtime+time.perf_counter()-trial_start
            with publication():
                if improved:
                    export(ema.ema,best,params,settings,0.,best_epoch)
                    # Same portable best checkpoint, also exposed as requested best.pt.
                    temp_best=work/'best.pt.tmp';shutil.copyfile(best,temp_best);temp_best.replace(work/'best.pt')
                save_json(work/'history.json',history)
                atomic_torch_save(checkpoint(epoch+1,elapsed),last)
                commit(epoch+1)
            print(f'Epoch {epoch+1}/{args.epochs} completed — checkpoint saved locally',flush=True)
            if getattr(args,'stop_after_epoch',0) and epoch+1>=args.stop_after_epoch:
                stopped(trial,epoch+1,'Explicit persistence smoke-test stop');return
            if stop_due() and epoch+1<args.epochs:
                stopped(trial,epoch+1,'Time reserve reached; final persistent upload follows')
                return
        del model,optimizer,ema,criterion,dl,batch,loss,weighted
        if device.type=='cuda': torch.cuda.empty_cache()
        if stop_due(): stopped(trial,args.epochs,'Full-board validation deferred to next session');return
        model,meta=load_model(best,str(device))
        co=read_json(data/'annotations/val_boards.json')
        predictions=predict_coco(model,co,data,settings,out/trial/'validation',0.,overlays=False,stop_check=check_budget)
        metrics,curve=evaluate_predictions(co,predictions)
        save_json(out/trial/'validation/metrics.json',metrics);save_json(out/trial/'validation/curves.json',curve)
        with publication():
            export(model,best,params,settings,metrics['threshold'],best_epoch)
            temp_best=work/'best.pt.tmp';shutil.copyfile(best,temp_best);temp_best.replace(work/'best.pt')
            commit(args.epochs)
        chosen=next(r for r in history['val'] if r['epoch']==best_epoch)
        stats.append(dict(config_name=trial,architecture='YOLOv8m-seg',seed=args.seed,epochs=args.epochs,
            total_params=params['total'],trainable_params=params['trainable'],
            train_loss=sum(chosen['train_epoch_loss_terms'].values()),val_loss=chosen['loss'],
            loss_epoch=best_epoch,val_metrics=metrics,test_metrics=None,threshold=metrics['threshold'],
            data_ratio=manifest['data_ratio'],runtime_seconds=runtime+time.perf_counter()-trial_start,
            checkpoint_name=str(best),settings=settings,split_fingerprint=protocol,
            optimizer='AdamW',augmentation=config['augmentation'],
            software=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,
                          python=platform.python_version(),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')))
        with publication(): save_json(stats_path,stats)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
        if stop_due():
            break
    selected=[s for s in stats if s['config_name'] in args.trials]
    complete=len(selected)==len(args.trials)
    save_json(out/'training_state.json',dict(complete=complete,validated_trials=[s['config_name'] for s in selected]))
    if not complete:
        print('Some trials remain. Resume training before test evaluation.',flush=True);return
    winner=max(selected,key=lambda s:(s['val_metrics']['S_seg'],-s['trainable_params']))
    save_json(out/'selection_frozen.json',dict(best_trial=winner['config_name'],test_used_for_selection=False,
        split_fingerprint=protocol,trials=[dict(config_name=s['config_name'],threshold=s['threshold'],
            checkpoint_name=s['checkpoint_name'],val_S_seg=s['val_metrics']['S_seg'],trainable_params=s['trainable_params']) for s in selected],
        selection_rule='Validation Graph-F reconstructed-instance S_seg; fewer trainable parameters breaks a tie'))
    with publication():
        target=out/'checkpoints/best_segment.ckpt';temporary=target.with_suffix('.tmp')
        shutil.copyfile(winner['checkpoint_name'],temporary);temporary.replace(target)
    print('Frozen best trial:',winner['config_name'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--trials',nargs='+',choices=list(TRIALS),default=list(TRIALS))
    p.add_argument('--weights',default='yolov8m-seg.pt')
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--hours',type=float,default=6.)
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-cpu',action='store_true',help='Verification only')
    p.add_argument('--backup-config',help='Public destination config only; credentials from environment/Secrets')
    p.add_argument('--shutdown-buffer-seconds',type=float,default=900)
    p.add_argument('--persistence-smoke',action='store_true',help='Separate synthetic reliability check; not assignment results')
    p.add_argument('--stop-after-epoch',type=int,default=0,help='Reliability smoke test only; keep 0 in actual trials')
    run(p.parse_args())
