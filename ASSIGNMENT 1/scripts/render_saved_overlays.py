"""Render deterministic test overlays from saved predictions, without new inference."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as masks

ROOT=Path(__file__).resolve().parents[1]
HEX='E53935 1E88E5 00ACC1 8E24AA FB8C00 43A047 6D4C41 D81B60 3949AB C0CA33 00897B FFB300 5E35B1 039BE5 757575 FDD835 212121 7E57C2 F4511E 2E7D32 283593 7CB342 546E7A C2185B F9A825'.split()
COLORS=np.array([[int(h[i:i+2],16) for i in [0,2,4]] for h in HEX],dtype=np.float32)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def run(archive, maximum=1800):
    manifest=read(ROOT/'data/segmentation_manifest.json')
    records=manifest['records']
    f=sorted([r for r in records if r['split']=='test' and r['domain']=='Graph-F'],key=lambda r:r['id'])[:2]
    w=sorted([r for r in records if r['split']=='test' and r['domain']=='Graph-W'],key=lambda r:r['id'])[:1]
    selected=f+w
    outputs=[]
    with zipfile.ZipFile(archive) as z:
        for name in ['yolov8m_640_aug','yolov8m_960_aug']:
            stats=next(r for r in read(ROOT/'segmentation_stats.json') if r['config_name']==name)
            predictions=read(ROOT/f'results/segmentation/{name}/fast_test/predictions.json')
            for record in selected:
                source=z.read(record['zip_image'])
                import hashlib
                if hashlib.sha256(source).hexdigest()!=record['sha256']:
                    raise ValueError('Source image differs from recorded manifest: '+record['zip_image'])
                image=Image.open(io.BytesIO(source)).convert('RGB')
                image.thumbnail((maximum,maximum),Image.Resampling.LANCZOS)
                canvas=np.array(image,dtype=np.float32)
                accepted=[p for p in predictions if p['image_id']==record['id'] and p['score']>=stats['threshold']]
                for p in sorted(accepted,key=lambda p:p['score']):
                    binary=masks.decode(p['segmentation'])
                    resized=np.array(Image.fromarray(binary).resize(image.size,Image.Resampling.NEAREST),dtype=bool)
                    canvas[resized]=.55*canvas[resized]+.45*COLORS[p['category_id']-1]
                final=Image.fromarray(canvas.astype(np.uint8))
                label=f"{name} | test board {record['id']} | t*={stats['threshold']:.2f} | {len(accepted)} masks"
                draw=ImageDraw.Draw(final)
                draw.rectangle([0,0,min(final.width,len(label)*7+12),24],fill='white')
                draw.text((5,5),label,fill='black')
                out=ROOT/f"results/segmentation/{name}_test_{record['id']:04d}.png"
                final.save(out)
                outputs.append({'file':out.relative_to(ROOT).as_posix(),'trial':name,'image_id':record['id'],
                                'domain':record['domain'],'source_image':record['zip_image'],'source_sha256':record['sha256'],
                                'selection_rule':'first two official Graph-F test IDs and first Graph-W test ID; no score-based selection',
                                'threshold':stats['threshold'],'display_size':list(final.size),'predicted_masks_above_threshold':len(accepted)})
                print('Rendered',out.name,flush=True)
    (ROOT/'results/segmentation/overlay_manifest.json').write_text(json.dumps(outputs,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--maximum-display-size',type=int,default=1800)
    a=parser.parse_args()
    run(a.archive,a.maximum_display_size)
