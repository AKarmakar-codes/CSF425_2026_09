"""Create the exact G09 archive only after the required artifact audit passes."""
import argparse
import json
from pathlib import Path
import zipfile
import sys
sys.dont_write_bytecode=True
from check_submission import audit, sha

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT.parent)
    a=p.parse_args()
    report=audit(ROOT,load_checkpoints=True)
    if not report['artifact_checks_pass']:
        raise SystemExit('Submission incomplete. Run scripts/check_submission.py for the missing files. No final archive created.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    target=a.output_dir/'CSF425_2026_G09_Assignment-I.zip'
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f in sorted(ROOT.rglob('*')):
            if f.is_file() and not any(part in {'__pycache__','.git'} for part in f.relative_to(ROOT).parts):
                arc=Path('CSF425_2026_G09')/f.relative_to(ROOT)
                info=zipfile.ZipInfo.from_file(f,arc.as_posix())
                if f.suffix=='.sh':
                    info.create_system=3
                    info.external_attr=(0o100755<<16)
                z.writestr(info,f.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:
            raise RuntimeError('ZIP CRC failure: '+bad)
    print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'sha256':sha(target)},indent=2))
