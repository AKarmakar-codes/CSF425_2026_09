"""Colab preflight: CUDA operators, real data, loss routing, and export/inference."""
import argparse
import copy
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import numpy as np
from PIL import Image
from mmcv.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import pseudo_collate
from train import make_config,export_checkpoint
from inference_segmentation import load_model,predict_board,read_json


def check(data):
    data=Path(data).resolve()
    assert torch.cuda.is_available(),'Enable a GPU runtime'
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
    _,keep=nms(boxes,torch.tensor([.9,.8],device='cuda'),.5)
    assert keep.numel()==1,'MMCV CUDA NMS failed'
    with tempfile.TemporaryDirectory() as tmp:
        cfg=make_config(data,Path(tmp),'tiny_640',epochs=5,workers=0)
        # Preflight initializes random weights; training itself uses COCO weights.
        cfg.load_from=None
        runner=Runner.from_cfg(cfg)
        runner._init_model_weights()
        model=runner.model
        model.eval()
        loader_cfg=copy.deepcopy(cfg.train_dataloader)
        loader_cfg.dataset.dataset.pipeline=copy.deepcopy(cfg.val_dataloader.dataset.pipeline)
        loader=Runner.build_dataloader(loader_cfg,seed=42)
        dataset=loader.dataset.dataset
        def example_index(domain):
            for i in range(len(dataset)):
                info=dataset.get_data_info(i)
                if domain+'__' in info['img_path'] and any(
                    a['bbox'][2]-a['bbox'][0]>48 and a['bbox'][3]-a['bbox'][1]>48
                    for a in info['instances']): return i
            raise ValueError('No suitable annotated '+domain+' smoke-test tile')
        fi=example_index('Graph-F')
        wi=example_index('Graph-W')
        batch=pseudo_collate([dataset[fi],dataset[wi]])
        batch=model.data_preprocessor(batch,training=True)
        weak_batch=copy.deepcopy(batch)
        losses=model(**batch,mode='loss')
        total=sum(sum(v) if isinstance(v,list) else v for v in losses.values())
        assert torch.isfinite(total),'Nonfinite initial loss'
        assert losses['loss_mask'].item()>0.,'Graph-F mask branch received no supervision'
        total.backward()
        assert model.bbox_head.rtm_cls[0].weight.grad is not None
        # The same real batch marked Graph-W must have zero mask loss.
        for sample in weak_batch['data_samples']:
            sample.set_metainfo(dict(img_path='Graph-W__preflight.jpg'))
        with torch.no_grad():
            weak_losses=model(**weak_batch,mode='loss')
        assert weak_losses['loss_mask'].item()==0.,'Graph-W mask-loss leakage'
        settings=dict(tile=1024,overlap=256,mask_nms_iou=.5,max_detections=5000,
                      score_floor=.001,mask_binary_threshold=.5)
        raw=Path(tmp)/'raw.pth'
        portable=Path(tmp)/'portable.ckpt'
        torch.save(dict(state_dict=model.state_dict(),meta={}),raw)
        params=dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        export_checkpoint(raw,portable,cfg,params,settings,.25)
        del losses,total,weak_losses,model,runner,batch,weak_batch
        torch.cuda.empty_cache()
        infer,_=load_model(portable)
        # Exercise image preprocessing, mask generation and RLE without GT at inference.
        image=np.zeros((160,224,3),np.uint8)
        predict_board(infer,image,1,settings)
        print('PREFLIGHT PASSED: CUDA NMS, finite losses, backward, zero W mask loss, portable inference.')
        print('25-class parameter count:',params)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    check(p.parse_args().data)
