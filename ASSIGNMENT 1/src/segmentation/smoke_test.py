"""Check the actual medium model, loss routing, backward, export and inference."""
import argparse
import copy
import tempfile
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data import CocoTiles,collate,to_device
from model import make_model,MixedMaskLoss,export
from inference_segmentation import load_model,predict_tile


def check(data=None,device=None):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model=make_model().to(device).eval()
    params=dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    size=128
    images=torch.rand(2,3,size,size,device=device)
    masks=torch.zeros(2,size,size,device=device,dtype=torch.long)
    masks[0,32:96,32:96]=1
    batch=dict(img=images,masks=masks,cls=torch.tensor([[0.],[1.]],device=device),
        bboxes=torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]],device=device),
        batch_idx=torch.tensor([0,1],device=device),mask_valid=torch.tensor([True,False],device=device))
    criterion=MixedMaskLoss(model)
    pred=model(images)
    loss,terms=criterion(pred,batch)
    assert torch.isfinite(loss).all() and terms[1]>0
    loss.sum().backward()
    assert any(p.grad is not None for p in model.model[-1].parameters() if p.requires_grad)
    weak=dict(batch,mask_valid=torch.tensor([False,False],device=device))
    with torch.no_grad():
        _,weak_terms=criterion(model(images),weak)
    assert weak_terms[1].item()==0.,'Graph-W placeholder masks contributed mask loss'
    # Changing only Graph-W mask pixels must not change the mask loss.
    altered=dict(batch,masks=masks.clone());altered['masks'][1]=1
    with torch.no_grad():
        _,original_terms=criterion(model(images),batch)
        _,altered_terms=criterion(model(images),altered)
    assert torch.allclose(original_terms[1],altered_terms[1],atol=1e-6)
    if device.type=='cuda':
        # Exercise the actual fused optimizer + AMP combination before a long run.
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,fused=True)
        scaler=torch.amp.GradScaler('cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            mixed_loss,_=criterion(model(images),batch)
        assert torch.isfinite(mixed_loss).all()
        scaler.scale(mixed_loss.sum()/len(images)).backward()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=before, 'AMP skipped the GPU optimizer preflight step'
        assert all(torch.isfinite(p).all() for p in model.parameters())
        optimizer.zero_grad(set_to_none=True)
        del optimizer,scaler
        print('PASS: CUDA AMP + fused AdamW optimizer step')
    if data:
        ds=CocoTiles(data,'train_tiles.json',128)
        fi=next(i for i,im in enumerate(ds.images) if im['mask_valid'] and ds.by_image[im['id']])
        wi=next(i for i,im in enumerate(ds.images) if not im['mask_valid'] and ds.by_image[im['id']])
        real=to_device(collate([ds[fi],ds[wi]]),device)
        with torch.no_grad(): real_loss,_=criterion(model(real['img']),real)
        assert torch.isfinite(real_loss).all()
    settings=dict(tile=1024,overlap=256,input_size=128,score_floor=.001,
                  max_per_tile=300,max_detections=5000,mask_nms_iou=.5,mask_binary_threshold=.5)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'test.ckpt'
        export(model,path,params,settings,.25,0)
        loaded,meta=load_model(path,str(device))
        with torch.no_grad():
            assert torch.allclose(model(images)[0],loaded(images)[0],atol=1e-6)
        predict_tile(loaded,np.zeros((80,112,3),np.uint8),settings)
    print('PASS: actual YOLOv8m-seg forward/backward, F mask loss, zero W mask loss, W-mask invariance, export/reload and inference')
    print('Exact 25-class parameter counts:',params)
    return params


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--device')
    a=p.parse_args();check(a.data,a.device)
