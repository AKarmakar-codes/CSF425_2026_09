"""Classification derivation; the shared segmentation manifest is never rewritten."""
import argparse
import sys
import copy
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.classification.preprocessing import contract, crop_rgb, bounds
from src.classification.provenance import digest, file_hash, new_run, write_json, ROOT

MAPPING = json.loads((ROOT / 'data_mapping.json').read_text(encoding='utf8'))
SYMBOLS = {c['symbol'].lower(): c['id'] for c in MAPPING['classes']}
TYPES = {k.lower().replace('_', ' '): v for k, v in MAPPING['source_mappings']['WACV_2019'].items()}
TYPES.update({'electrolytic capacitor': 2, 'zener diode': 8, 'resistor network': 9,
              'resistor coil': 7, 'ferrite bead': 16, 'emi filter': 16,
              'integrated circuit': 11, 'jumper link': 25, 'jumper': 25,
              'qa': 24, 'clock': -1, 'display': -1, 'switch (s)': 23,
              'switch (sw)': 18, 'vacuum tube': 21, 'plug': 12, 'motor': 14,
              'thyristor': 13, 'light emitting diode': 22,
              'resistor jumper': -1, 'diode zener array': -1})

def map_graph_w_name(raw_name):
    raw = ' '.join(str(raw_name).lower().replace('"', '').replace("'", '').replace('_', ' ').split())
    # A fully specified canonical name is authoritative, including U versus IC.
    for c in MAPPING['classes']:
        if raw == c['name'].lower():
            return c['id']
    for name in sorted(TYPES, key=lambda x: (-len(x), x)):
        if raw == name or raw.startswith(name + ' '):
            cid = TYPES[name]
            return cid if cid > 0 else None
    match = re.fullmatch(r'([a-z]+)(?:\d+)?', raw)
    return SYMBOLS.get(match[1]) if match else None

def normalize_board_id(value):
    value = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    value = re.sub(r'\.(pt|xml|jpg|jpeg|png)$', '', value, flags=re.I)
    value = re.sub(r'^graph_(?:\d+_)?', '', value, flags=re.I)
    return re.sub(r'_(mask|image)$', '', value, flags=re.I).casefold()

def exact_split(board_id, train_names, test_names):
    tr, te = ({normalize_board_id(n) for n in names} for names in (train_names, test_names))
    if tr & te:
        raise ValueError('Official split overlap')
    key = normalize_board_id(board_id)
    if key not in tr | te:
        raise ValueError(f'Board missing from official partitions: {board_id}')
    return 'train' if key in tr else 'test'

def membership(manifest):
    result = {}
    for split in ['train', 'val', 'test']:
        for b in manifest[split + '_boards']:
            key = b['dataset'] + '/' + normalize_board_id(b['board_id'])
            if key in result:
                raise ValueError(f'Duplicate/overlapping board {key}')
            result[key] = split
    return result

def verify_official(manifest, source_root):
    status = {}
    for ds in ['Graph-W', 'Graph-F']:
        paths = {s: Path(source_root) / ds / 'graphs' / s for s in ['train', 'test']}
        if not all(p.is_dir() for p in paths.values()):
            status[ds] = 'unverified: official graph lists unavailable'
            continue
        names = {s: {normalize_board_id(p.name) for p in d.glob('*.pt')} for s, d in paths.items()}
        observed = set()
        for s in ['train', 'val', 'test']:
            for b in manifest[s + '_boards']:
                if b['dataset'] != ds:
                    continue
                actual = exact_split(b['board_id'], names['train'], names['test'])
                if actual != ('test' if s == 'test' else 'train'):
                    raise ValueError(f'Legacy/official split mismatch: {ds}/{b["board_id"]}')
                observed.add(normalize_board_id(b['board_id']))
        if observed != names['train'] | names['test']:
            raise ValueError(f'Official partition completeness mismatch for {ds}')
        status[ds] = 'verified exact complete disjoint membership'
    return status

def read_xml(path):
    rows = []
    for obj in ET.parse(path).getroot().findall('object'):
        box = obj.find('bndbox')
        rows.append({'raw_name': obj.findtext('name'),
                     'bbox': [float(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]})
    return rows

def derive(legacy_path, output_dir=None, source_root='data/GraphPCB', img_size=224, cache=False):
    legacy_path = Path(legacy_path)
    old = json.loads(legacy_path.read_text(encoding='utf8'))
    frozen = membership(old)
    official = verify_official(old, source_root)
    out, run_id = new_run(output_dir, 'manifest')
    result = {s + suffix: [] for s in ['train', 'val', 'test'] for suffix in ['_boards', '_crops']}
    counts, transitions, examples, availability = Counter(), Counter(), defaultdict(list), Counter()
    annotation_hashes = {}
    for split in ['train', 'val', 'test']:
        for board in old[split + '_boards']:
            b = copy.deepcopy(board)
            ds, bid = b['dataset'], b['board_id']
            image = Path(source_root) / ds / 'image_dir' / Path(b['image_path']).name
            b['image_path'] = str(image.resolve())
            xml = Path(source_root) / ds / 'mask_dir' / (bid + '.xml')
            use_xml = ds == 'Graph-W' and xml.is_file()
            anns = read_xml(xml) if use_xml else b['annotations']
            annotation_hashes[ds + '/' + bid] = file_hash(xml) if use_xml else digest(anns)
            availability['source_xml_boards' if use_xml else 'legacy_annotation_boards'] += 1
            before = {(tuple(a['bbox']), a.get('raw_name')): a['class_id'] for a in b['annotations']}
            b['annotations'] = []
            img = Image.open(image).convert('RGB') if image.is_file() else None
            availability['images_present' if img else 'images_missing'] += 1
            image_hash = file_hash(image) if img else None
            if cache and img is None:
                raise FileNotFoundError(f'Cannot cache missing image: {image}')
            for i, a in enumerate(anns):
                raw = a.get('raw_name')
                prior = before.get((tuple(a['bbox']), raw))
                cid = map_graph_w_name(raw) if ds == 'Graph-W' and raw is not None else a.get('class_id')
                if cid is not None and not 1 <= cid <= 25:
                    raise ValueError(f'Invalid class {cid}')
                key = f'{prior if prior is not None else "unknown"}->{cid if cid is not None else "ignore"}'
                transitions[key] += 1
                if len(examples[key]) < 8 and raw not in examples[key]:
                    examples[key].append(raw)
                counts[str(cid) if cid else 'ignore'] += 1
                if cid is None:
                    continue
                if img:
                    bounds(a['bbox'], img.size)
                else:
                    x1, y1, x2, y2 = a['bbox']
                    if not (0 <= x1 < x2 and 0 <= y1 < y2):
                        raise ValueError('Invalid annotation bounds')
                a = dict(a, class_id=cid)
                b['annotations'].append(a)
                sample_id = ds + '/' + bid + '/' + str(i)
                sample = dict(sample_id=sample_id, board_id=bid, dataset=ds, image_path=b['image_path'],
                              bbox=a['bbox'], class_id=cid, raw_name=raw, storage='source', source_image_sha256=image_hash,
                              annotation_sha256=annotation_hashes[ds+'/'+bid])
                if cache:
                    identity = digest({'image': image_hash, 'annotation': annotation_hashes[ds+'/'+bid],
                                       'bbox': a['bbox'], 'contract': contract(img_size)})
                    cp = out / 'cache' / (identity + '.png')
                    cp.parent.mkdir(exist_ok=True)
                    if not cp.exists():
                        crop_rgb(img, a['bbox']).save(cp, 'PNG')
                    sample.update(storage='cache', crop_path=str(cp), cache_id=identity, crop_sha256=file_hash(cp))
                result[split + '_crops'].append(sample)
            result[split + '_boards'].append(b)
    assert membership(result) == frozen
    meta = {'schema': 'classification-manifest-v2', 'run_id': run_id, 'parent_manifest_sha256': file_hash(legacy_path),
            'mapping_implementation_sha256': file_hash(__file__),
            'taxonomy_sha256': digest(MAPPING['classes']), 'mapping_sha256': file_hash(ROOT/'data_mapping.json'),
            'preprocessing': contract(img_size), 'preprocessing_sha256': digest(contract(img_size)),
            'annotation_sha256': digest(annotation_hashes), 'annotation_hashes': annotation_hashes, 'official_membership': official,
            'validation_membership': 'frozen from original manifest', 'availability': dict(availability),
            'completeness': 'source XML available for some boards' if availability['source_xml_boards'] else 'legacy retained annotations only; ignored source objects unrecoverable'}
    result['metadata'] = meta
    report = {'counts_after': dict(counts), 'transitions': dict(transitions), 'representative_raw_labels': dict(examples),
              'before_counts': dict(Counter(str(a['class_id']) for s in ['train','val','test'] for b in old[s+'_boards'] for a in b['annotations'])),
              'metadata': meta, 'guidance': 'Project data_mapping.json declares potentiometer->1, ic->11; external source-label documentation unavailable. clock/display unresolved ignore; no test performance used.'}
    write_json(out / 'manifest.json', result)
    write_json(out / 'mapping_audit.json', report)
    return out / 'manifest.json'

def validate_manifest(path, require_images=True):
    m = json.loads(Path(path).read_text(encoding='utf8'))
    membership(m)
    if m.get('metadata', {}).get('schema') != 'classification-manifest-v2':
        raise ValueError('Repaired classification manifest required; derive from legacy first')
    if m['metadata']['taxonomy_sha256'] != digest(MAPPING['classes']):
        raise ValueError('Taxonomy mismatch')
    if m['metadata']['mapping_sha256'] != file_hash(ROOT/'data_mapping.json'):
        raise ValueError('Mapping changed; regenerate classification manifest/cache')
    if m['metadata'].get('mapping_implementation_sha256') != file_hash(__file__):
        raise ValueError('Mapping implementation changed or is unknown; regenerate classification manifest/cache')
    if digest(m['metadata']['preprocessing']) != m['metadata']['preprocessing_sha256']:
        raise ValueError('Preprocessing identity mismatch')
    ids = set()
    for s in ['train', 'val', 'test']:
        expected = [(b, a) for b in m[s+'_boards'] for a in b['annotations']]
        rows = m[s+'_crops']
        if len(rows) != len(expected):
            raise ValueError('Crop/annotation count mismatch')
        for row, (b, a) in zip(rows, expected):
            if row['sample_id'] in ids or not 1 <= row['class_id'] <= 25:
                raise ValueError('Duplicate sample or invalid label')
            ids.add(row['sample_id'])
            if row['image_path'] != b['image_path'] or row['bbox'] != a['bbox'] or row['class_id'] != a['class_id'] or row['board_id'] != b['board_id']:
                raise ValueError('Crop/board annotation mismatch')
            if row['storage'] == 'cache':
                expected_id = digest({'image':row['source_image_sha256'], 'annotation':row['annotation_sha256'],
                                      'bbox':row['bbox'], 'contract':m['metadata']['preprocessing']})
                if expected_id != row['cache_id']:
                    raise ValueError('Stale cache identity; regenerate')
            if require_images:
                from src.classification.dataset import load_sample_image
                load_sample_image(row)
    return m

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-manifest', default='data/dataset_manifest.json')
    p.add_argument('--source-root', default='data/GraphPCB')
    p.add_argument('--output-dir')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--cache', action='store_true')
    p.add_argument('--validate')
    p.add_argument('--allow-missing-images', action='store_true')
    a = p.parse_args()
    if a.validate:
        m = validate_manifest(a.validate, not a.allow_missing_images)
        print(json.dumps({'valid': True, 'samples': {s: len(m[s+'_crops']) for s in ['train','val','test']}, 'images_checked': not a.allow_missing_images}))
    else:
        print(derive(a.legacy_manifest, a.output_dir, a.source_root, a.img_size, a.cache))

if __name__ == '__main__':
    main()
