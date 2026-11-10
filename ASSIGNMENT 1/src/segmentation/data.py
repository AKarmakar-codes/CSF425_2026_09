"""Exact COCO-mask loading and synchronized single-image augmentation for YOLO."""
import collections
import json
import hashlib
import os
from pathlib import Path
import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mu
from scipy.ndimage import find_objects

import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from inference_segmentation import letterbox


def transform(image,idmap,boxes,size,rng,augment):
    """Keep mask IDs/holes exact through nearest-neighbor geometric transforms."""
    canvas,gain,left,top,nw,nh=letterbox(image,size)
    resized=cv2.resize(idmap,(nw,nh),interpolation=cv2.INTER_NEAREST)
    output_ids=np.zeros((size,size),np.uint16)
    output_ids[top:top+nh,left:left+nw]=resized
    boxes=boxes.copy()
    if len(boxes): boxes=boxes*gain+np.array([left,top,left,top])
    if augment:
        angle=float(rng.uniform(-15,15));scale=float(rng.uniform(.75,1.25))
        matrix=cv2.getRotationMatrix2D((size/2,size/2),angle,scale)
        matrix[:,2]+=rng.uniform(-.1,.1,2)*size
        canvas=cv2.warpAffine(canvas,matrix,(size,size),flags=cv2.INTER_LINEAR,
                              borderValue=(114,114,114))
        output_ids=cv2.warpAffine(output_ids,matrix,(size,size),flags=cv2.INTER_NEAREST,borderValue=0)
        if len(boxes):
            corners=boxes[:,[0,1,2,1,2,3,0,3]].reshape(-1,4,2)
            moved=corners@matrix[:,:2].T+matrix[:,2]
            boxes=np.concatenate([moved.min(1),moved.max(1)],axis=1)
        if rng.random()<.5:
            canvas=canvas[:,::-1].copy();output_ids=output_ids[:,::-1].copy()
            if len(boxes): boxes[:,[0,2]]=size-boxes[:,[2,0]]
        if rng.random()<.5:
            canvas=canvas[::-1].copy();output_ids=output_ids[::-1].copy()
            if len(boxes): boxes[:,[1,3]]=size-boxes[:,[3,1]]
        hsv=cv2.cvtColor(canvas,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0]=(hsv[:,:,0]+rng.uniform(-3,3))%180
        hsv[:,:,1]=np.clip(hsv[:,:,1]*rng.uniform(.8,1.2),0,255)
        hsv[:,:,2]=np.clip(hsv[:,:,2]*rng.uniform(.8,1.2),0,255)
        canvas=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    return canvas,output_ids,np.clip(boxes,0,size)


class CocoTiles(Dataset):
    def __init__(self,data,annotation,size,augment=False,seed=42,balance=False,cache_masks=True):
        self.root=Path(data);self.size=size;self.augment=augment;self.seed=seed;self.epoch=0
        annotation_text=(self.root/'annotations'/annotation).read_text()
        self.co=json.loads(annotation_text)
        # Cache only exact, unaugmented integer masks. Changes to annotations
        # invalidate this cache; random transforms are still applied on every use.
        self.mask_cache=None
        if cache_masks:
            signature=hashlib.sha256(annotation_text.encode()).hexdigest()[:20]
            self.mask_cache=self.root/'mask_cache_v1'/signature
            self.mask_cache.mkdir(parents=True,exist_ok=True)
        self.images=self.co['images'];self.by_image=collections.defaultdict(list)
        for a in self.co['annotations']: self.by_image[a['image_id']].append(a)
        counts=collections.Counter(c for im in self.images for c in
                                   {a['category_id'] for a in self.by_image[im['id']]})
        self.indices=[]
        for i,im in enumerate(self.images):
            classes={a['category_id'] for a in self.by_image[im['id']]}
            repeat=max([1.]+[(.01/(counts[c]/len(self.images)))**.5 for c in classes]) if balance else 1.
            self.indices.extend([i]*int(np.ceil(repeat)))

    def __len__(self): return len(self.indices)

    def __getitem__(self,index):
        epoch=self.epoch
        if isinstance(index,tuple): epoch,index=index
        im=self.images[self.indices[index]]
        image=cv2.imread(str(self.root/im['file_name']))
        if image is None: raise FileNotFoundError(im['file_name'])
        anns=self.by_image[im['id']]
        if len(anns)>=65535: raise ValueError('Too many instance IDs for uint16 augmentation')
        cached=self.mask_cache/f'{im["id"]}.npy' if self.mask_cache and im['mask_valid'] else None
        idmap=np.load(cached,allow_pickle=False) if cached is not None and cached.exists() else None
        cache_miss=idmap is None
        if cache_miss: idmap=np.zeros(image.shape[:2],np.uint16)
        boxes=[];labels=[]
        for i,a in enumerate(anns,1):
            x,y,w,h=a['bbox'];boxes.append([x,y,x+w,y+h]);labels.append(a['category_id']-1)
            if im['mask_valid'] and cache_miss:
                mask=mu.decode(a['segmentation']).astype(bool)
                if np.any(idmap[mask]): raise ValueError('Unexpected overlap in recovered Graph-F instances')
                idmap[mask]=i
        if cached is not None and cache_miss:
            # Repeated samples may be handled concurrently by different workers.
            temporary=cached.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as f: np.save(f,idmap,allow_pickle=False)
            temporary.replace(cached)
        boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,index]))
        image,idmap,boxes=transform(image,idmap,boxes,self.size,rng,self.augment)
        # One linear scan replaces an entire HxW scan for EACH instance.
        regions=find_objects(idmap,max_label=len(anns)) if im['mask_valid'] and anns else []
        remap=np.zeros(len(anns)+1,dtype=np.uint16);keep_boxes=[];keep_labels=[]
        for i,(box,label) in enumerate(zip(boxes,labels),1):
            if im['mask_valid']:
                region=regions[i-1]
                if region is None: continue
                yy,xx=region
                box=np.array([xx.start,yy.start,xx.stop,yy.stop],dtype=np.float32)
            if box[2]<=box[0] or box[3]<=box[1]: continue
            keep_boxes.append(box);keep_labels.append(label)
            if im['mask_valid']: remap[i]=len(keep_boxes)
        new_ids=remap[idmap]
        boxes=np.asarray(keep_boxes,dtype=np.float32).reshape(-1,4)
        if len(boxes):
            wh=boxes[:,2:]-boxes[:,:2]
            boxes=np.concatenate([boxes[:,:2]+wh/2,wh],axis=1)/self.size
        return dict(img=torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))),
                    cls=torch.tensor(keep_labels,dtype=torch.float32).view(-1,1),
                    bboxes=torch.from_numpy(boxes),masks=torch.from_numpy(new_ids.astype(np.int64)),
                    mask_valid=bool(im['mask_valid']),image_id=im['id'])


def collate(samples):
    return dict(img=torch.stack([s['img'] for s in samples]),
        cls=torch.cat([s['cls'] for s in samples]),bboxes=torch.cat([s['bboxes'] for s in samples]),
        masks=torch.stack([s['masks'] for s in samples]),
        batch_idx=torch.cat([torch.full((len(s['cls']),),i,dtype=torch.int64) for i,s in enumerate(samples)]),
        mask_valid=torch.tensor([s['mask_valid'] for s in samples],dtype=torch.bool))


def to_device(batch,device):
    batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
    batch['img']=batch['img'].float()/255.
    return batch
