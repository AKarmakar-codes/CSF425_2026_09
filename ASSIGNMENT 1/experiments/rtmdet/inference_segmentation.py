"""Standalone construction, tiled inference, mask metrics and visualization.

Only Python's standard library and installed external packages are imported.
The exported checkpoint embeds the full inference config; no project import,
training data, MMDetection source checkout, or separate config is needed.
"""
import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ('R','C','U','Q','J','L','RA','D','RN','TP','IC','P','CR','M',
           'BTN','FB','CRA','SW','T','F','V','LED','S','QA','JP')
HEX = ['E53935','1E88E5','00ACC1','8E24AA','FB8C00','43A047','6D4C41',
       'D81B60','3949AB','C0CA33','00897B','FFB300','5E35B1','039BE5',
       '757575','FDD835','212121','7E57C2','F4511E','2E7D32','283593',
       '7CB342','546E7A','C2185B','F9A825']


def save_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rle_json(rle):
    return dict(size=list(rle['size']),counts=rle['counts'].decode('ascii')
                if isinstance(rle['counts'],bytes) else rle['counts'])


def place_mask(mask,x,y,height,width):
    xx,yy=np.nonzero(mask.T)
    indices=(xx.astype(np.int64)+x)*height+yy+y
    if not len(indices):
        return rle_json(mu.frPyObjects(dict(size=[height,width],
                                          counts=[height*width]),height,width))
    gaps=np.flatnonzero(np.diff(indices)!=1)
    starts=indices[np.r_[0,gaps+1]]
    ends=indices[np.r_[gaps,len(indices)-1]]
    counts=[int(starts[0])]
    for i,(a,b) in enumerate(zip(starts,ends)):
        if i: counts.append(int(a-ends[i-1]-1))
        counts.append(int(b-a+1))
    counts.append(int(height*width-ends[-1]-1))
    return rle_json(mu.frPyObjects(dict(size=[height,width],counts=counts),height,width))


def tile_starts(length,tile,overlap):
    if length<=tile: return [0]
    starts=list(range(0,length-tile+1,tile-overlap))
    if starts[-1]!=length-tile: starts.append(length-tile)
    return starts


def load_model(checkpoint,device='cuda:0'):
    import torch
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.utils import register_all_modules
    from mmcv.transforms import Compose
    register_all_modules(init_default_scope=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'pcb' not in ckpt.get('meta',{}):
        raise ValueError('Use an exported checkpoint from this notebook, with meta.pcb.')
    meta=ckpt['meta']['pcb']
    cfg=Config(copy.deepcopy(meta['inference_config']))
    # Avoid downloading initialization weights during standalone inference.
    cfg.model.backbone.init_cfg=None
    cfg.model.bbox_head.type='RTMDetInsSepBNHead'
    cfg.pop('custom_imports',None)
    model=init_detector(cfg,str(checkpoint),device=device,palette='random')
    model.dataset_meta=dict(classes=CLASSES,palette=[tuple(int(h[i:i+2],16)
                           for i in (0,2,4)) for h in HEX])
    pipeline=[copy.deepcopy(t) for t in cfg.test_pipeline if t['type']!='LoadAnnotations']
    pipeline[0]['type']='mmdet.LoadImageFromNDArray'
    model._pcb_pipeline=Compose(pipeline)
    return model,meta


def suppress_duplicates(predictions,iou_threshold=.5,max_detections=5000):
    """Score-ordered, class-aware mask NMS in original board coordinates."""
    kept=[]
    by_class={}
    for p in sorted(predictions,key=lambda d:d['score'],reverse=True):
        cls=by_class.setdefault(p['category_id'],[])
        x,y,w,h=p['bbox']
        candidates=[q for q in cls if x<q['bbox'][0]+q['bbox'][2] and
                    q['bbox'][0]<x+w and y<q['bbox'][1]+q['bbox'][3] and
                    q['bbox'][1]<y+h]
        if candidates:
            overlaps=mu.iou([p['segmentation']],[q['segmentation'] for q in candidates],
                            [0]*len(candidates))
            if float(np.max(overlaps))>iou_threshold: continue
        cls.append(p)
        kept.append(p)
        if len(kept)>=max_detections: break
    return kept


def predict_board(model,image,image_id,settings):
    """Return COCO RLE masks; never materialize N full-board masks on the GPU."""
    from mmdet.apis import inference_detector
    h,w=image.shape[:2]
    tile=settings['tile']
    xs=tile_starts(w,tile,settings['overlap'])
    ys=tile_starts(h,tile,settings['overlap'])
    all_predictions=[]
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            crop=image[y:min(y+tile,h),x:min(x+tile,w)]
            pred=inference_detector(model,crop,test_pipeline=model._pcb_pipeline).pred_instances.cpu()
            masks=pred.masks.numpy()
            scores=pred.scores.numpy()
            labels=pred.labels.numpy()
            # Voronoi ownership at tile centers removes most seam duplicates.
            left=0 if ix==0 else (xs[ix-1]+tile+x)/2
            right=w if ix==len(xs)-1 else (x+tile+xs[ix+1])/2
            top=0 if iy==0 else (ys[iy-1]+tile+y)/2
            bottom=h if iy==len(ys)-1 else (y+tile+ys[iy+1])/2
            for mask,score,label in zip(masks,scores,labels):
                yy,xx=np.nonzero(mask[:crop.shape[0],:crop.shape[1]])
                if not len(xx): continue
                x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
                cx,cy=x+(x0+x1)/2,y+(y0+y1)/2
                if not (left<=cx<right and top<=cy<bottom): continue
                binary=mask[y0:y1,x0:x1].astype(np.uint8)
                gx,gy=x+x0,y+y0
                all_predictions.append(dict(image_id=int(image_id),category_id=int(label)+1,
                    score=float(score),bbox=[gx,gy,x1-x0,y1-y0],
                    segmentation=place_mask(binary,gx,gy,h,w),
                    local_mask=rle_json(mu.encode(np.asfortranarray(binary))),
                    mask_origin=[gx,gy]))
            del pred,masks
    return suppress_duplicates(all_predictions,settings['mask_nms_iou'],settings['max_detections'])


def overlay(image,predictions,path,threshold,max_side=1800):
    h,w=image.shape[:2]
    scale=min(1.,max_side/max(h,w))
    canvas=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
    for p in sorted(predictions,key=lambda d:d['score']):
        if p['score']<threshold: continue
        rgb=tuple(int(HEX[p['category_id']-1][i:i+2],16) for i in (0,2,4))
        color=rgb[::-1]
        mask=mu.decode(p['local_mask'])
        x,y=p['mask_origin']
        xa,ya=round(x*scale),round(y*scale)
        xb,yb=round((x+mask.shape[1])*scale),round((y+mask.shape[0])*scale)
        xb,yb=min(xb,canvas.shape[1]),min(yb,canvas.shape[0])
        if xb<=xa or yb<=ya: continue
        small=cv2.resize(mask,(xb-xa,yb-ya),interpolation=cv2.INTER_NEAREST).astype(bool)
        region=canvas[ya:yb,xa:xb]
        region[small]=(region[small]*.55+np.array(color)*.45).astype(np.uint8)
        contours,_=cv2.findContours(small.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(region,contours,-1,color,1)
        cv2.putText(canvas,f"{CLASSES[p['category_id']-1]} {p['score']:.2f}",
                    (xa,max(10,ya)),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1,cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),canvas)


def clean_predictions(predictions,kind='segm'):
    keys=('image_id','category_id','score','segmentation') if kind=='segm' else \
         ('image_id','category_id','score','bbox')
    # Remove bbox for segm: COCO's area ranges must use mask area, not box area.
    return [{k:p[k] for k in keys} for p in predictions]


def coco_metrics(ground_truth,predictions,kind='segm',max_dets=100):
    gt=COCO()
    gt.dataset=copy.deepcopy(ground_truth)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if predictions:
            dt=gt.loadRes(clean_predictions(predictions,kind))
        else:
            dt=COCO()
            dt.dataset=dict(images=gt.dataset['images'],categories=gt.dataset['categories'],annotations=[])
            dt.createIndex()
        ev=COCOeval(gt,dt,kind)
        ev.params.maxDets=[1,10,max_dets]
        ev.evaluate()
        ev.accumulate()
    precision=ev.eval['precision'][:,:,:,0,-1]  # T,R,K,area=all,maxDets
    def avg(v):
        valid=v[v>=0]
        return float(valid.mean()) if valid.size else None
    ap=avg(precision)
    ap50=avg(precision[0])
    per_class=[]
    for i,c in enumerate(CLASSES):
        per_class.append(dict(id=i+1,code=c,AP=avg(precision[:,:,i]),AP50=avg(precision[0,:,i])))
    pr=precision[0].copy()
    valid=pr>=0
    mean_pr=np.divide(np.where(valid,pr,0).sum(1),valid.sum(1),
                      out=np.zeros(pr.shape[0]),where=valid.sum(1)>0)
    return dict(AP50=ap50,mAP_50_95=ap,per_class=per_class,max_dets=max_dets,
                iou_type=kind,recall_grid=ev.params.recThrs.tolist(),
                precision_at_iou50=mean_pr.tolist())


def mask_f1_curves(ground_truth,predictions,thresholds=None):
    """Micro instance F1: same-class greedy matching at mask IoU >= .50.

    Score ordering means all thresholds can share a single matching pass.
    Macro F1 averages the fixed 25 classes with zero for undefined classes.
    """
    thresholds=np.linspace(0,1,101) if thresholds is None else np.asarray(thresholds)
    gt_by=collections_by_key(ground_truth['annotations'])
    pr_by=collections_by_key(predictions)
    entries=[]
    gt_counts=np.zeros(25,dtype=int)
    for a in ground_truth['annotations']: gt_counts[a['category_id']-1]+=1
    for key,preds in pr_by.items():
        targets=gt_by.get(key,[])
        preds=sorted(preds,key=lambda p:p['score'],reverse=True)
        used=set()
        overlaps=mu.iou([p['segmentation'] for p in preds],
                        [g['segmentation'] for g in targets],[0]*len(targets)) if targets else None
        for i,p in enumerate(preds):
            hit=False
            if targets:
                possible=[j for j in range(len(targets)) if j not in used]
                if possible:
                    j=max(possible,key=lambda j:overlaps[i,j])
                    if overlaps[i,j]>=.5:
                        used.add(j)
                        hit=True
            entries.append((p['score'],p['category_id']-1,hit))
    result=dict(thresholds=thresholds.tolist(),precision=[],recall=[],f1=[],macro_f1=[],per_class=[])
    for t in thresholds:
        tp=np.zeros(25,dtype=int)
        fp=np.zeros(25,dtype=int)
        for score,cid,hit in entries:
            if score>=t:
                if hit: tp[cid]+=1
                else: fp[cid]+=1
        fn=gt_counts-tp
        denom=2*tp+fp+fn
        f=np.divide(2*tp,denom,out=np.zeros(25),where=denom>0)
        result['precision'].append(float(tp.sum()/max(1,(tp+fp).sum())))
        result['recall'].append(float(tp.sum()/max(1,gt_counts.sum())))
        result['f1'].append(float(2*tp.sum()/max(1,denom.sum())))
        result['macro_f1'].append(float(f.mean()))
        result['per_class'].append([dict(id=i+1,code=c,F1=float(f[i]),TP=int(tp[i]),
            FP=int(fp[i]),FN=int(fn[i]),support=int(gt_counts[i])) for i,c in enumerate(CLASSES)])
    return result


def collections_by_key(items):
    groups={}
    for item in items:
        groups.setdefault((item['image_id'],item['category_id']),[]).append(item)
    return groups


def subset_coco(co,domain=None,masks_only=False):
    imgs=[im for im in co['images'] if (domain is None or im['domain']==domain)
          and (not masks_only or im['mask_valid'])]
    ids={im['id'] for im in imgs}
    return dict(co,images=imgs,annotations=[a for a in co['annotations'] if a['image_id'] in ids])


def evaluate_predictions(co,predictions,threshold=None):
    fco=subset_coco(co,masks_only=True)
    fids={im['id'] for im in fco['images']}
    fp=[p for p in predictions if p['image_id'] in fids]
    ap=coco_metrics(fco,fp)
    dense=coco_metrics(fco,fp,max_dets=1000)
    curve=mask_f1_curves(fco,fp)
    # Largest threshold breaks a tie, reducing false positives.
    idx=max(range(101),key=lambda i:(curve['f1'][i],curve['thresholds'][i])) if threshold is None \
        else int(np.argmin(np.abs(np.asarray(curve['thresholds'])-threshold)))
    selected=curve['thresholds'][idx]
    if ap['AP50'] is None or ap['mAP_50_95'] is None:
        raise ValueError('No evaluable Graph-F validation/test masks')
    result=dict(mask_scope='Graph-F only; connected-component instance reconstruction',
        AP50=ap['AP50'],mAP_50_95=ap['mAP_50_95'],F1_mask=curve['f1'][idx],
        F1_mask_macro25=curve['macro_f1'][idx],threshold=selected,
        S_seg=.2*ap['AP50']+.5*ap['mAP_50_95']+.3*curve['f1'][idx],
        class_AP=ap['per_class'],class_F1=curve['per_class'][idx],
        standard_COCO=ap,dense_COCO_max1000=dense,
        all_boards_mask_metrics=None,
        graph_w_mask_metrics=None,
        graph_w_reason='No pixel ground truth in supplied ZIP',
        bbox_all=coco_metrics(co,predictions,kind='bbox'))
    for dom in ('Graph-F','Graph-W'):
        dco=subset_coco(co,domain=dom)
        ids={im['id'] for im in dco['images']}
        if ids:
            result['bbox_'+dom]=coco_metrics(dco,[p for p in predictions if p['image_id'] in ids],kind='bbox')
    return result,curve


def predict_coco(model,co,data_root,settings,out_dir,threshold,overlays=True):
    predictions=[]
    out_dir=Path(out_dir)
    for i,im in enumerate(co['images'],1):
        # PIL handles long Windows paths and non-ASCII filenames reliably.
        image=np.array(Image.open(Path(data_root)/im['file_name']).convert('RGB'))[:,:,::-1].copy()
        start=time.perf_counter()
        p=predict_board(model,image,im['id'],settings)
        predictions.extend(p)
        if overlays:
            overlay(image,p,out_dir/'overlays'/f"{im['id']:04d}_{Path(im['file_name']).stem}.png",threshold)
        print(f"[{i}/{len(co['images'])}] {im['file_name']}: {len(p)} predictions, "
              f"{time.perf_counter()-start:.1f}s",flush=True)
    save_json(out_dir/'predictions.json',predictions)
    return predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--images',help='Image file or directory; no annotations required')
    p.add_argument('--coco',help='Optional COCO annotation file for evaluation')
    p.add_argument('--data-root',default='.')
    p.add_argument('--out',default='results/segmentation/inference')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    model,meta=load_model(args.checkpoint,args.device)
    settings=meta['settings']
    threshold=meta['threshold']
    if args.coco:
        co=read_json(args.coco)
        preds=predict_coco(model,co,args.data_root,settings,args.out,threshold)
        metrics,curve=evaluate_predictions(co,preds,threshold)
        save_json(Path(args.out)/'metrics.json',metrics)
        save_json(Path(args.out)/'curves.json',curve)
    else:
        if not args.images: p.error('Provide --images or --coco')
        root=Path(args.images)
        paths=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
        if not paths: raise FileNotFoundError('No input images')
        all_preds=[]
        for image_id,path in enumerate(paths,1):
            image=np.array(Image.open(path).convert('RGB'))[:,:,::-1].copy()
            preds=predict_board(model,image,image_id,settings)
            overlay(image,preds,Path(args.out)/f'{image_id:04d}_{path.stem}.png',threshold)
            all_preds.extend(preds)
        save_json(Path(args.out)/'predictions.json',all_preds)
    print('Frozen confidence threshold:',threshold)


if __name__=='__main__':
    main()
