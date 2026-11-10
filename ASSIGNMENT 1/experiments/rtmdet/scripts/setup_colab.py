"""Install an isolated Python 3.10 environment, independent of Colab's kernel."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def setup(env):
    if sys.platform!='linux':
        raise RuntimeError('Run this setup script inside Google Colab / Linux with an NVIDIA GPU')
    env=Path(env)
    uv=shutil.which('uv')
    if uv is None:
        subprocess.run([sys.executable,'-m','pip','install','uv==0.8.22'],check=True)
        uv=shutil.which('uv')
    if uv is None: raise RuntimeError('uv executable was not installed on PATH')
    subprocess.run([uv,'python','install','3.10'],check=True)
    if not (env/'bin/python').exists():
        subprocess.run([uv,'venv','--python','3.10','--seed',str(env)],check=True)
    py=str(env/'bin/python')
    def pip(*args):
        subprocess.run([uv,'pip','install','--python',py,*args],check=True)
    pip('torch==2.1.2','torchvision==0.16.2',
        '--index-url','https://download.pytorch.org/whl/cu121')
    pip('numpy==1.26.4','setuptools==69.5.1','wheel==0.43.0',
        'mmengine==0.10.7','opencv-python==4.10.0.84',
        'scipy==1.11.4','matplotlib==3.8.4','pycocotools==2.0.7',
        'PyYAML==6.0.2','tqdm==4.66.5','psutil==6.0.0','yapf==0.40.1')
    pip('mmcv==2.1.0','--no-deps','--only-binary=:all:',
        '--find-links','https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html')
    pip('mmdet==3.3.0')
    check="""
import torch, torchvision, mmcv, mmengine, mmdet, numpy
from mmcv.ops import nms
assert numpy.__version__ == '1.26.4'
assert mmcv.__version__ == '2.1.0'
assert torch.cuda.is_available(), 'In Colab choose Runtime > Change runtime type > T4 GPU (or another NVIDIA GPU)'
x=torch.tensor([[0.,0.,10.,10.]],device='cuda')
nms(x,torch.tensor([1.],device='cuda'),.5)
print('Environment OK:',torch.__version__,torch.version.cuda,mmcv.__version__,mmdet.__version__,torch.cuda.get_device_name(0))
"""
    subprocess.run([py,'-c',check],check=True)
    print('Use this interpreter for every project command:',py)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--env',default='/content/pcb-venv')
    setup(p.parse_args().env)
