"""Select a candidate using only source-validation macro-F1 and enforce cap."""
import argparse, json, shutil
from pathlib import Path

CAP=23_600_000
def main(args):
 rows=[]
 for name in args.runs:
  root=Path(name);stats=json.loads((root/'semantic_stats.json').read_text())
  f1=stats.get('val_metrics',{}).get('macro_f1');threshold=stats.get('threshold');params=stats.get('params_inference')
  if f1 is None or threshold is None or params is None:raise ValueError(f'Incomplete source validation stats: {root}')
  if params>CAP:raise ValueError(f'Over-budget candidate {root}: {params} > {CAP}')
  rows.append({'run_id':root.name,'run_dir':str(root.resolve()),'model_config':stats.get('model_config'),'source_val_macro_f1':f1,'source_val_macro_iou':stats['val_metrics'].get('macro_iou'),'source_val_loss':stats['val_loss'],'threshold':threshold,'params_inference':params,'params_trainable':stats['params_trainable'],'runtime_seconds':stats['runtime_seconds']})
 if not rows:raise ValueError('No candidate runs supplied')
 split_ids={json.loads((Path(r['run_dir'])/'semantic_stats.json').read_text()).get('split_id') for r in rows}
 seeds={json.loads((Path(r['run_dir'])/'semantic_stats.json').read_text()).get('seed') for r in rows}
 if len(split_ids)!=1 or None in split_ids or len(seeds)!=1:raise ValueError('Candidate runs must use one identical source board split and seed')
 rows.sort(key=lambda r:(-r['source_val_macro_f1'],-(r['source_val_macro_iou'] or 0),r['params_inference'],r['source_val_loss']))
 selected=rows[0];doc={'selection_rule':'highest source-validation semantic macro-F1; ties by macro-mIoU, lower parameter count, then lower validation loss','target_data_used_for_selection':False,'parameter_ceiling':CAP,'selected':selected,'candidates':rows}
 out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(doc,indent=2,allow_nan=False))
 if args.promote_dir:
  dest=Path(args.promote_dir);dest.mkdir(parents=True,exist_ok=True);src=Path(selected['run_dir'])
  for task in ('semantic','instance'):shutil.copy2(src/f'best_{task}.ckpt',dest/f'best_{task}.ckpt')
  (dest/'selection.json').write_text(json.dumps(doc,indent=2,allow_nan=False))
 print(json.dumps(doc,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--promote-dir');main(p.parse_args())
