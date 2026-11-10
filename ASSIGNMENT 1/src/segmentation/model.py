"""Official YOLOv8m-seg model; source-aware mask loss adds no parameters."""
import copy
from types import SimpleNamespace
from pathlib import Path
import torch
import os
import uuid
import ultralytics
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.loss import v8SegmentationLoss


class MixedMaskLoss(v8SegmentationLoss):
    def __call__(self,preds,batch):
        self.mask_valid=batch['mask_valid'].to(self.device)
        return super().__call__(preds,batch)

    def calculate_segmentation_loss(self,fg_mask,masks,target_gt_idx,target_bboxes,
                                   batch_idx,proto,pred_masks,imgsz,overlap):
        valid_fg=fg_mask & self.mask_valid[:,None]
        if not valid_fg.any():
            return proto.reshape(-1)[:1].sum()*0 + pred_masks.reshape(-1)[:1].sum()*0
        # Detection losses were already computed from the original fg_mask.
        return super().calculate_segmentation_loss(valid_fg,masks,target_gt_idx,
            target_bboxes,batch_idx,proto,pred_masks,imgsz,overlap)


def make_model(weights=None,freeze=0):
    if ultralytics.__version__!='8.3.161': raise RuntimeError('Expected ultralytics==8.3.161')
    model=SegmentationModel('yolov8m-seg.yaml',nc=25,verbose=False)
    if model.yaml.get('scale')!='m': raise RuntimeError('Expected medium scale, not nano/small')
    if weights:
        from ultralytics import YOLO
        model.load(YOLO(str(weights)).model,verbose=False)
    args=copy.deepcopy(DEFAULT_CFG_DICT)
    args.update(box=7.5,cls=.5,dfl=1.5,overlap_mask=True)
    model.args=SimpleNamespace(**args)
    for index,module in enumerate(model.model):
        if index<freeze:
            for p in module.parameters(): p.requires_grad=False
    return model


def set_train_mode(model,freeze):
    model.train()
    for i,module in enumerate(model.model):
        if i<freeze: module.eval()  # freeze BN running statistics too


def export(model,path,params,settings,threshold,epoch):
    from inference_segmentation import CLASSES
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    meta=dict(model_yaml=copy.deepcopy(model.yaml),settings=settings,threshold=threshold,
              parameters=params,classes=list(CLASSES),epoch=epoch,
              architecture='YOLOv8m-seg; CSP-style Conv/C2f/SPPF backbone',
              mask_supervision='Graph-F masks only; Graph-W class and box supervision',
              ultralytics_version=ultralytics.__version__)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('wb') as f:
        torch.save(dict(format='graphpcb-yolov8m-seg-v1',meta=meta,state_dict=state),f)
        f.flush();os.fsync(f.fileno())
    temporary.replace(path)
