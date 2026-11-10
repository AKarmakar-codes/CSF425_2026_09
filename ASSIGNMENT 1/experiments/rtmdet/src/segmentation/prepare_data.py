"""Convert the supplied GraphPCB ZIP without unpickling graph files.

Graph-F instances are recovered from connected regions in the semantic masks.
Graph-W has boxes only: its rectangles are API placeholders, never mask targets.
"""
import argparse
import collections
import hashlib
import io
import json
import re
import shlex
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu

CLASSES = ('R','C','U','Q','J','L','RA','D','RN','TP','IC','P','CR','M',
           'BTN','FB','CRA','SW','T','F','V','LED','S','QA','JP')
SOURCE_RGB = [(255,0,0),(255,255,0),(0,234,255),(170,0,255),(255,127,0),
              (191,255,0),(0,149,255),(106,255,0),(0,64,255),(237,185,185),
              (185,215,237),(231,233,185),(220,185,237),(185,237,224),
              (143,35,35),(35,98,143),(143,106,35),(107,35,143),(79,143,35),
              (115,115,115),(204,204,204),(245,130,48),(220,190,255),
              (170,255,195),(255,250,200)]
DISPLAY_HEX = ['#E53935','#1E88E5','#00ACC1','#8E24AA','#FB8C00','#43A047',
               '#6D4C41','#D81B60','#3949AB','#C0CA33','#00897B','#FFB300',
               '#5E35B1','#039BE5','#757575','#FDD835','#212121','#7E57C2',
               '#F4511E','#2E7D32','#283593','#7CB342','#546E7A','#C2185B',
               '#F9A825']
# Exact source type, parsed with shlex to preserve quoted multiword names.
W_MAP = {'resistor':'R', 'capacitor':'C', 'electrolytic capacitor':'C',
         'ic':'IC', 'transistor':'Q', 'connector':'J', 'inductor':'L',
         'diode':'D', 'zener diode':'D', 'diode zener array':'D',
         'resistor network':'RN', 'test point':'TP', 'button':'BTN',
         'ferrite bead':'FB', 'switch':'SW', 'fuse':'F', 'led':'LED',
         'jumper':'JP'}
PALETTE_SOURCE = 'https://github.com/CandleLabAI/PCBSegClassNet/blob/main/src/data/create_mask.py'


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf8')


def json_rle(rle):
    return {'size': list(rle['size']), 'counts': rle['counts'].decode('ascii')
            if isinstance(rle['counts'], bytes) else rle['counts']}


def region_rle(crop, x, y, height, width):
    """Place a small binary crop in a full-size RLE, without allocating H*W."""
    yy, xx = np.nonzero(crop.T)  # transpose: first index is the original x
    if not len(xx):
        return json_rle(mu.frPyObjects({'size':[height,width],
                                        'counts':[height*width]}, height, width))
    positions = (yy.astype(np.int64) + x) * height + xx + y
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, len(positions)-1]]
    counts = [int(starts[0])]
    for i, (start, end) in enumerate(zip(starts, ends)):
        if i:
            counts.append(int(start - ends[i-1] - 1))
        counts.append(int(end - start + 1))
    counts.append(int(height*width - ends[-1] - 1))
    return json_rle(mu.frPyObjects({'size':[height,width], 'counts':counts},
                                   height, width))


def positions(length, tile, overlap):
    if length <= tile:
        return [0]
    starts = list(range(0, length-tile+1, tile-overlap))
    if starts[-1] != length-tile:
        starts.append(length-tile)
    return starts


def f_instances(rgb):
    """Keep every connected component, including single-pixel regions."""
    packed = (rgb[:,:,0].astype(np.uint32)<<16) | \
             (rgb[:,:,1].astype(np.uint32)<<8) | rgb[:,:,2].astype(np.uint32)
    lut = np.full(1<<24, 255, np.uint8)
    lut[0] = 0
    for k, (r,g,b) in enumerate(SOURCE_RGB, 1):
        lut[(r<<16)|(g<<8)|b] = k
    semantic = lut[packed]
    if np.any(semantic == 255):
        unknown = np.unique(packed[semantic == 255])
        raise ValueError(f'Unknown Graph-F mask colors: {unknown[:20].tolist()}. '
                         'Do not guess a nearest color or a class mapping.')
    objects = []
    for cid in np.unique(semantic):
        if cid == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (semantic == cid).astype(np.uint8), connectivity=8)
        for instance_id in range(1, n):
            x,y,w,h,area = map(int, stats[instance_id])
            crop = (labels[y:y+h,x:x+w] == instance_id).astype(np.uint8)
            objects.append((int(cid), x,y,w,h,crop, int(area)))
    return objects


def w_instances(xml_bytes, width, height):
    objects, ignored = [], collections.Counter()
    for obj in ET.fromstring(xml_bytes).findall('object'):
        name = obj.findtext('name', '')
        try:
            source_type = shlex.split(name)[0].lower()
        except (ValueError, IndexError):
            source_type = name.lower()
        code = W_MAP.get(source_type)
        if code is None:
            ignored[source_type] += 1
            continue
        box = obj.find('bndbox')
        # Pascal VOC inclusive, 1-based -> half-open 0-based pixel coordinates.
        x1=max(0, int(float(box.findtext('xmin')))-1)
        y1=max(0, int(float(box.findtext('ymin')))-1)
        x2=min(width, int(float(box.findtext('xmax'))))
        y2=min(height, int(float(box.findtext('ymax'))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'Invalid XML box for {name}')
        crop = np.ones((y2-y1, x2-x1), np.uint8)
        objects.append((CLASSES.index(code)+1, x1,y1,x2-x1,y2-y1,crop,
                        (x2-x1)*(y2-y1)))
    return objects, ignored


def group_key(domain, stem):
    if domain == 'Graph-F':
        return re.match(r'pcb_(\d+)', stem).group(1)
    # Conservative grouping of front/back/top/bottom views from the same board.
    return re.sub(r'[_-](top|bottom).*$', '', stem, flags=re.I)


def inventory(z):
    records = []
    for domain, expected in [('Graph-F',(115,47)), ('Graph-W',(37,10))]:
        for split, count in zip(('train','test'), expected):
            graphs = sorted(n for n in z.namelist()
                            if f'/{domain}/graphs/{split}/' in n and n.endswith('.pt'))
            if len(graphs) != count:
                raise ValueError(f'{domain}/{split}: expected {count}, got {len(graphs)}')
            for graph in graphs:
                stem = re.sub(r'^graph_\d+_', '', Path(graph).stem)
                prefix = graph.split('/graphs/')[0]
                image = prefix+'/image_dir/'+stem+('_image.png' if domain=='Graph-F' else '.jpg')
                ann = prefix+'/mask_dir/'+stem+('_mask.png' if domain=='Graph-F' else '.xml')
                if image not in z.namelist() or ann not in z.namelist():
                    raise FileNotFoundError((image,ann))
                records.append(dict(domain=domain, official_split=split, stem=stem,
                                    group=domain+':'+group_key(domain,stem),
                                    zip_image=image, zip_annotation=ann))
    return records


def training_class_presence(z, records):
    """Scan official training labels only; reserve rare classes for training."""
    presence=collections.defaultdict(set)
    color_to_code={tuple(color):code for color,code in zip(SOURCE_RGB,CLASSES)}
    for i,r in enumerate([r for r in records if r['official_split']=='train'],1):
        if r['domain']=='Graph-F':
            im=Image.open(io.BytesIO(z.read(r['zip_annotation']))).convert('RGB')
            colors=im.getcolors(maxcolors=4096)
            if colors is None:
                raise ValueError('Unexpected large color palette: '+r['stem'])
            codes=set()
            for _,color in colors:
                if color==(0,0,0): continue
                if color not in color_to_code:
                    raise ValueError(f'Unknown source mask color {color} in {r["stem"]}')
                codes.add(color_to_code[color])
        else:
            codes=set()
            for obj in ET.fromstring(z.read(r['zip_annotation'])).findall('object'):
                try: source=shlex.split(obj.findtext('name',''))[0].lower()
                except (ValueError,IndexError): continue
                if source in W_MAP: codes.add(W_MAP[source])
        r['training_source_class_codes']=sorted(codes)
        presence[r['group']].update(codes)
        if i%25==0: print(f'Scanned class support in {i}/152 official training boards',flush=True)
    return presence


def make_split(records, seed=42, val_fraction=.2, class_presence=None):
    rng = np.random.default_rng(seed)
    val_groups = set()
    for domain in ('Graph-F','Graph-W'):
        groups = sorted({r['group'] for r in records
                         if r['domain']==domain and r['official_split']=='train'})
        count=max(1,round(len(groups)*val_fraction))
        if class_presence is None:
            selected=rng.choice(groups,count,replace=False).tolist()
        else:
            # A group containing the last remaining training example of a class
            # cannot be assigned to validation. No official test labels are read.
            remaining=collections.Counter(c for g in groups for c in class_presence[g])
            selected=[]
            for g in rng.permutation(groups).tolist():
                if all(remaining[c]>1 for c in class_presence[g]):
                    selected.append(g)
                    remaining.subtract(class_presence[g])
                if len(selected)==count: break
            if not selected:
                raise ValueError('Cannot form validation while retaining training class support')
        val_groups.update(selected)
    for r in records:
        r['split'] = ('test' if r['official_split']=='test'
                      else 'val' if r['group'] in val_groups else 'train')
    return records


def coco_base():
    return dict(info={'description':'GraphPCB; see data_mapping.json for limitations'},
                licenses=[], images=[], annotations=[],
                categories=[dict(id=i+1,name=c) for i,c in enumerate(CLASSES)])


def annotation(objects, image_id, height, width, domain):
    out=[]
    for cid,x,y,w,h,crop,area in objects:
        ann=dict(image_id=image_id,category_id=cid,bbox=[x,y,w,h],area=area,iscrowd=0)
        # W placeholders are only needed by the shared MMDetection loader.
        ann['segmentation']=region_rle(crop,x,y,height,width)
        ann['mask_valid'] = domain=='Graph-F'
        out.append(ann)
    return out


def prepare(zip_path, out, tile=1024, overlap=256, seed=42, limit=None):
    out=Path(out)
    if not 0 <= overlap < tile:
        raise ValueError('Require 0 <= overlap < tile')
    if (out/'manifest.json').exists():
        raise FileExistsError(f'{out}/manifest.json already exists; use a new output directory')
    z=zipfile.ZipFile(zip_path)
    signature=hashlib.sha256(json.dumps([(i.filename,i.CRC,i.file_size)
                            for i in z.infolist()]).encode()).hexdigest()
    records=inventory(z)
    presence=training_class_presence(z,records)
    records=make_split(records,seed,class_presence=presence)
    counts=collections.Counter((r['domain'],r['split']) for r in records)
    print('Board counts:', dict(counts), flush=True)
    full={s:coco_base() for s in ('train','val','test')}
    tiles={s:coco_base() for s in ('train','val')}
    ignored=collections.Counter()
    seen_hashes={}
    selected=records if limit is None else records[:limit]
    for image_id,r in enumerate(selected,1):
        raw=z.read(r['zip_image'])
        sha=hashlib.sha256(raw).hexdigest()
        if sha in seen_hashes and seen_hashes[sha]['official_split'] != r['official_split']:
            raise ValueError('Exact image duplicate across official train/test partitions')
        seen_hashes[sha]=r
        image=np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
        h,w=image.shape[:2]
        image_rel=Path('boards')/r['domain']/Path(r['zip_image']).name
        (out/image_rel).parent.mkdir(parents=True,exist_ok=True)
        (out/image_rel).write_bytes(raw)
        if r['domain']=='Graph-F':
            mask=np.array(Image.open(io.BytesIO(z.read(r['zip_annotation']))).convert('RGB'))
            if mask.shape != image.shape:
                raise ValueError('Image/mask shape mismatch')
            objects=f_instances(mask)
        else:
            objects,ign=w_instances(z.read(r['zip_annotation']),w,h)
            if r['official_split']=='train':
                ignored.update(ign)
        r.update(id=image_id,file_name=image_rel.as_posix(),width=w,height=h,
                 sha256=sha,mask_valid=r['domain']=='Graph-F')
        split=r['split']
        full[split]['images'].append({k:r[k] for k in
                                     ('id','file_name','width','height','domain','mask_valid')})
        full[split]['annotations'].extend(annotation(objects,image_id,h,w,r['domain']))
        if split!='test':
            for y in positions(h,tile,overlap):
                for x in positions(w,tile,overlap):
                    th,tw=min(tile,h-y),min(tile,w-x)
                    clipped=[]
                    for cid,ox,oy,ow,oh,crop,area in objects:
                        xa,ya=max(x,ox),max(y,oy)
                        xb,yb=min(x+tw,ox+ow),min(y+th,oy+oh)
                        if xb<=xa or yb<=ya:
                            continue
                        part=crop[ya-oy:yb-oy,xa-ox:xb-ox]
                        if not part.any():
                            continue
                        # Tighten the box after clipping; preserve every nonzero pixel.
                        yy,xx=np.nonzero(part)
                        ax,ay,bx,by=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
                        clipped.append((cid,xa-x+ax,ya-y+ay,bx-ax,by-ay,
                                        part[ay:by,ax:bx],int(part.sum())))
                    tid=len(tiles[split]['images'])+1
                    rel=Path('tiles')/split/f"{r['domain']}__{r['stem']}__{x}_{y}.jpg"
                    (out/rel).parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(image[y:y+th,x:x+tw]).save(out/rel,quality=95)
                    tiles[split]['images'].append(dict(id=tid,file_name=rel.as_posix(),
                        width=tw,height=th,domain=r['domain'],mask_valid=r['mask_valid'],
                        board_id=image_id))
                    tiles[split]['annotations'].extend(annotation(clipped,tid,th,tw,r['domain']))
        print(f"[{image_id}/{len(selected)}] {r['domain']} {r['stem']}: {len(objects)} regions",flush=True)
    for kind, sets in [('boards',full),('tiles',tiles)]:
        for split,co in sets.items():
            for aid,a in enumerate(co['annotations'],1):
                a['id']=aid
            dump(out/'annotations'/f'{split}_{kind}.json',co)
            if kind=='tiles' and split=='val':
                ids={im['id'] for im in co['images'] if im['mask_valid']}
                fco=dict(co,images=[im for im in co['images'] if im['id'] in ids],
                         annotations=[a for a in co['annotations'] if a['image_id'] in ids])
                dump(out/'annotations'/'val_tiles_f.json',fco)
    mapping=dict(canonical_classes=[dict(id=i+1,code=c,source_rgb=SOURCE_RGB[i],
                                        display_hex=DISPLAY_HEX[i]) for i,c in enumerate(CLASSES)],
        source_palette=PALETTE_SOURCE,graph_w_type_mapping=W_MAP,
        graph_w_ignored_train_types=dict(ignored),
        graph_f_instance_rule='8-connected regions per exact RGB class; touching same-class objects remain merged; disconnected fragments remain separate',
        graph_w_mask_rule='Rectangles are loader placeholders. Mask loss is disabled for Graph-W. No Graph-W mask evaluation is valid.',
        graph_w_ambiguities='ic -> IC, switch -> SW. U versus IC and S versus SW remain distinct; ambiguous capacitor jumper, clock, pins, text, unknown etc are ignored.',
        input_images='Stored pixel values retained. No inferred HLS/RGB reversal.',
        external_pcb_images=0)
    dump(out/'data_mapping.json',mapping)
    support={}
    for domain in ('Graph-F','Graph-W'):
        support[domain]={split:sorted({c for r in records if r['domain']==domain and r['split']==split
                           for c in r.get('training_source_class_codes',[])}) for split in ('train','val')}
    manifest=dict(seed=seed,tile=tile,overlap=overlap,records=selected,archive_signature=signature,
                  training_class_support=support,
                  official_train_boards=152,official_test_boards=57,
                  distinct_training_boards=sum(r['split']=='train' for r in selected),
                  data_ratio=sum(r['split']=='train' for r in selected)/152,
                  debug_subset=limit is not None,
                  limitation='Graph-W lacks pixel masks. Graph-F instance identities are recovered, not provided.')
    dump(out/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--tile',type=int,default=1024)
    p.add_argument('--overlap',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int,help='Debug only; never use for final results')
    a=p.parse_args()
    prepare(a.zip,a.out,a.tile,a.overlap,a.seed,a.limit)
