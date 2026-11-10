"""Package source and generated segmentation outputs, excluding datasets."""
import argparse
import shutil
import zipfile
from pathlib import Path


def package(project,results,destination):
    project,results,destination=Path(project),Path(results),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in project.rglob('*'):
            if p.is_file() and not any(x in p.parts for x in ('__pycache__','.git','results','checkpoints')):
                z.write(p,Path('RTMDet_PCB_Colab')/p.relative_to(project))
        for p in results.rglob('*'):
            if not p.is_file(): continue
            rel=p.relative_to(results)
            # Training optimizer snapshots can be much larger than inference checkpoints.
            if 'work_dirs' in rel.parts and p.suffix in ('.pth','.ckpt'): continue
            target=(Path('RTMDet_PCB_Colab')/rel if rel.parts[0]=='checkpoints'
                    else Path('RTMDet_PCB_Colab/results/segmentation')/rel)
            z.write(p,target)
    print('Created segmentation package:',destination)
    print('This is not the combined assignment submission; classification is still required.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',required=True)
    p.add_argument('--results',required=True)
    p.add_argument('--destination',required=True)
    a=p.parse_args();package(a.project,a.results,a.destination)
