"""Tiny explicitly synthetic train/validation fixture; contains no test images."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.classification.provenance import write_json
from src.data_prep import derive

def assets(directory):
    import yaml
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=False)
    m = {s+k: [] for s in ['train','val','test'] for k in ['_boards','_crops']}
    source = d/'source'
    images = source/'Graph-W'/'image_dir'
    images.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for split in ['train','val']:
        for i in range(2):
            bid = split + str(i)
            image = images/(bid+'.png')
            Image.fromarray(rng.integers(0,256,(40,40,3),dtype=np.uint8)).save(image)
            ann = {'class_id': i+1, 'bbox': [1.2,2.1,38.4,39.], 'raw_name': ['resistor R1','capacitor C1'][i]}
            m[split+'_boards'].append({'dataset':'Graph-W','board_id':bid,'image_path':str(image),'annotations':[ann]})
    legacy = d/'synthetic_source_manifest.json'
    write_json(legacy, m)
    manifest = derive(legacy, d/'derived', source, img_size=32, cache=True)
    trial = dict(backbone='efficientnet_b0', pretrained=False, img_size=32, batch_size=2, epochs=1,
                 learning_rate=.0003, weight_decay=.01, loss='focal', focal_gamma=2., scheduler='cosine',
                 warmup_epochs=1, imbalance='combined', num_workers=0)
    cfg = d/'config.yml'
    cfg.write_text(yaml.safe_dump({'random_seed':42, 'classification':{'trials':{'synthetic_smoke':trial}}}),encoding='utf8')
    return manifest, cfg
