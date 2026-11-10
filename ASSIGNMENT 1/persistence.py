"""Verified, immutable epoch-boundary backups. No credentials are serialized here."""
import hashlib
import base64
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import nullcontext
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath


def utc(): return datetime.now(timezone.utc).isoformat()


def sanitize_cli_output(value):
    """Redact before truncating or serializing subprocess output/exception text."""
    if isinstance(value,bytes): value=value.decode('utf8',errors='replace')
    value=str(value or '')
    for name,secret in os.environ.items():
        if secret and re.search(r'TOKEN|PASSWORD|SECRET|CREDENTIAL|API_KEY|KAGGLE_KEY',name,re.I):
            value=value.replace(secret,'[REDACTED]')
    value=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',value)
    value=re.sub(r'(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9_./+=-]+','[REDACTED AUTH]',value)
    value=re.sub(r'KGAT_[A-Za-z0-9_-]+','[REDACTED TOKEN]',value)
    value=re.sub(r'''(?ix)(["']?(?:authorization|api[_-]?token|api[_-]?key|kaggle_key|key|password|access_token|secret_access_key)["']?\s*[:=]\s*)(["']?)[^\s,"'}]+''',
                 r'\1[REDACTED]',value)
    # Signed download URLs and embedded URL credentials must never reach logs.
    value=re.sub(r'https?://[^\s<>"\']+',lambda m:re.sub(r'//[^/@]+@','//[REDACTED]@',m[0].split('?')[0])+('?[REDACTED QUERY]' if '?' in m[0] else ''),value)
    return value[:16000]+('\n[output truncated]' if len(value)>16000 else '')


class KaggleCLIError(RuntimeError):
    def __init__(self,record):
        self.record=record
        super().__init__(f'Kaggle {record["operation"]} failed: returncode={record["returncode"]}, '
                         f'elapsed={record["elapsed_seconds"]:.3f}s; see sanitized CLI diagnostics')


class KagglePublicationMissingSnapshot(RuntimeError): pass


def transport_name(zip_name):
    safe_relative(zip_name)
    if '/' in zip_name or not re.fullmatch(r'(training|probe)_[\w]+\.zip',zip_name):
        raise ValueError('Invalid backup ZIP name')
    return zip_name+'.pcb64'


def encode_transport(source,destination):
    """ASCII armor prevents Kaggle archive extraction, including magic sniffing.

    Renaming a ZIP or wrapping it in another archive is not a preservation
    guarantee. The server receives no ZIP signature in this top-level object.
    """
    source,destination=Path(source),Path(destination)
    verify_bundle(source)
    header=dict(format='graphpcb-zip-base64-v1',zip_name=source.name,
                zip_size=source.stat().st_size,zip_sha256=sha256(source))
    with source.open('rb') as src,destination.open('wb') as dst:
        dst.write(json.dumps(header,sort_keys=True).encode()+b'\n')
        for block in iter(lambda:src.read(3*256*1024),b''): dst.write(base64.b64encode(block)+b'\n')
    return header


def decode_transport(source,destination):
    """Recover the original ZIP bytes, not a newly recompressed equivalent ZIP."""
    destination=Path(destination);temporary=destination.with_name(destination.name+'.decoding.tmp')
    try:
        with Path(source).open('rb') as src,temporary.open('wb') as dst:
            header=json.loads(src.readline(8192))
            if header.get('format')!='graphpcb-zip-base64-v1': raise ValueError('Unknown ZIP transport')
            if header['zip_name']!=destination.name: raise ValueError('Transport ZIP name mismatch')
            transport_name(header['zip_name']);h=hashlib.sha256();size=0
            while True:
                line=src.readline(1024*1024+2)
                if not line: break
                if len(line)>1024*1024+1: raise ValueError('Oversized transport line')
                block=base64.b64decode(line.rstrip(b'\r\n'),validate=True)
                dst.write(block);h.update(block);size+=len(block)
            if size!=header['zip_size'] or h.hexdigest()!=header['zip_sha256']:
                raise ValueError('Transport ZIP size/SHA-256 mismatch')
            dst.flush();os.fsync(dst.fileno())
        verify_bundle(temporary)
        temporary.replace(destination)
        return header
    finally: temporary.unlink(missing_ok=True)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def validate_local_commits(run):
    run=Path(run);stats=run/'results/segmentation/segmentation_stats.json'
    done={s['config_name'] for s in json.loads(stats.read_text())} if stats.exists() else set()
    for commit in (run/'results/segmentation/work_dirs').glob('*/commit.json'):
        row=json.loads(commit.read_text())
        for name,digest in row.get('published_hashes',{}).items():
            if row['trial'] in done and name.endswith('/last.pt'): continue
            path=run/Path(*safe_relative(name).parts)
            if not path.exists() or sha256(path)!=digest:
                raise ValueError('Incomplete or corrupt local publication: '+name)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def safe_relative(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe backup member')
    return p


def verify_bundle(bundle,expected_run=None):
    """Validate EVERY file before extracting or loading trusted torch state."""
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate archive members')
        for name in names: safe_relative(name)
        manifest=json.loads(z.read('backup_manifest.json'))
        if manifest.get('format')!='graphpcb-backup-v1': raise ValueError('Unknown backup format')
        if expected_run and manifest['run_name']!=expected_run: raise ValueError('Backup run name mismatch')
        files=manifest['files']
        if set(names)!=set(files)|{'backup_manifest.json'}: raise ValueError('Missing or unlisted backup files')
        for name,meta in files.items():
            h=hashlib.sha256();size=0
            with z.open(name) as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block);size+=len(block)
            if size!=meta['size'] or h.hexdigest()!=meta['sha256']: raise ValueError('Backup integrity failure: '+name)
        for item in manifest.get('checkpoints',[]):
            if item['path'] not in files: raise ValueError('Checkpoint missing from manifest')
        return manifest


def build_bundle(run,workspace,lock=None,kind='training'):
    """Copy a consistent committed generation under the trainer's publication lock.

    Compression/hashing/network happen after releasing it. GPU objects are never
    accessed by the worker. Incomplete epochs are deliberately excluded.
    """
    run,workspace=Path(run),Path(workspace);workspace.mkdir(parents=True,exist_ok=True)
    ident=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    bundle=workspace/f'{kind}_{ident}.zip'
    with tempfile.TemporaryDirectory(dir=workspace,prefix='snapshot_') as d:
        stage=Path(d)
        with lock if lock is not None else nullcontext():
            results=run/'results/segmentation'
            commits=sorted(results.glob('work_dirs/*/commit.json'))
            completed=[]
            stats=results/'segmentation_stats.json'
            done={s['config_name'] for s in json.loads(stats.read_text())} if stats.exists() else set()
            paths=[]
            # Small reproduction metadata, not datasets, credentials or upload config.
            for name in ('run_settings.json','manifest.json','data_mapping.json','environment_freeze.txt','smoke_fixture.json'):
                if (run/name).exists(): paths.append(run/name)
            for folder in [run/'source',*run.glob('source_upgrade_*')]:
                if folder.is_dir(): paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
            for commit in commits:
                row=json.loads(commit.read_text());trial=row['trial']
                last=commit.parent/'last.pt'
                if trial not in done and not last.exists(): raise ValueError('Committed checkpoint missing')
                if trial not in done: paths.append(last)
                completed.append(dict(trial=trial,completed_epoch=row['epoch'],best_validation_metric=row['best_ap'] if row['best_ap']>=0 else None,
                                      path=last.relative_to(run).as_posix(),finished=trial in done))
            # Legacy runs without commit records can be bootstrapped by the trainer,
            # but must never be silently backed up as an empty training snapshot.
            if kind=='training' and not commits: raise ValueError('No committed epoch checkpoint available yet')
            if results.exists():
                paths.extend(p for p in results.rglob('*') if p.is_file() and
                    (p.suffix=='.json' or p.suffix=='.ckpt' or p.name=='best.pt') and
                    'tile_validation' not in p.parts and 'test' not in p.parts)
            for path in sorted(set(paths)):
                rel=path.relative_to(run);dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
            # Completed trials need their portable best, but not their optimizer state.
            for row in completed:
                if row['finished']: row['path']=f'results/segmentation/checkpoints/{row["trial"]}.ckpt'
        files={p.relative_to(stage).as_posix():dict(size=p.stat().st_size,sha256=sha256(p))
               for p in stage.rglob('*') if p.is_file()}
        manifest=dict(format='graphpcb-backup-v1',kind=kind,run_name=run.name,created_utc=utc(),
            recovery='Last completed epoch only; repeat any interrupted epoch; not exact mid-epoch recovery',
            checkpoints=completed,files=files)
        # A process exception between local replacements must not publish a
        # mismatched last/best generation as a consistent remote snapshot.
        for commit in (stage/'results/segmentation/work_dirs').glob('*/commit.json'):
            row=json.loads(commit.read_text())
            for name,digest in row.get('published_hashes',{}).items():
                if name in files and files[name]['sha256']!=digest: raise ValueError('Incomplete local checkpoint publication')
        atomic_json(stage/'backup_manifest.json',manifest)
        temporary=bundle.with_suffix('.tmp')
        # Stored ZIP avoids spending training CPU on compressing large float tensors.
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED,allowZip64=True) as z:
            for p in stage.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(stage).as_posix())
        temporary.replace(bundle)
    verify_bundle(bundle,run.name)
    return bundle


def restore_bundle(bundle,run):
    run=Path(run);manifest=verify_bundle(bundle,run.name)
    if manifest['kind']!='training': raise ValueError('Not a training backup')
    # Do not mix a remote checkpoint with an already-started local experiment.
    if list((run/'results/segmentation/work_dirs').glob('*/last.pt')):
        raise FileExistsError('Local training state already exists; refusing to overwrite it')
    with zipfile.ZipFile(bundle) as z:
        for name in manifest['files']:
            target=run/Path(*safe_relative(name).parts);target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.restore.tmp')
            with z.open(name) as source,temp.open('wb') as dest:
                shutil.copyfileobj(source,dest,8*1024*1024);dest.flush();os.fsync(dest.fileno())
            temp.replace(target)
    for row in manifest['checkpoints']:
        if not row['finished']:
            print(f'Restored {row["trial"]} from completed epoch {row["completed_epoch"]}; next epoch {row["completed_epoch"]+1}',flush=True)
    return manifest


class KaggleStore:
    """Dedicated EXISTING private Dataset. Uses supported CLI version/download.

    Each new version carries the new snapshot and one verified predecessor.
    Old Dataset versions are never deleted. No /kaggle/input writes are used.
    """
    def __init__(self,config,workspace):
        self.config=config;self.slug=config['dataset'];self.workspace=Path(workspace)
        if not re.fullmatch(r'[\w-]+/[\w-]+',self.slug): raise ValueError('Set dataset to owner/private-dataset-slug')
        self.timeout=config.get('operation_timeout_seconds',600)
        self.cli=shutil.which('kaggle')
        if not self.cli: raise RuntimeError('Install kaggle before persistent backup setup')
        self.cache=self.workspace/'kaggle_cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.diagnostics_path=Path(config.get('diagnostics_path',self.workspace/'kaggle_cli_diagnostics.jsonl'))
        self._diagnostics_lock=threading.Lock()
        self.last_dataset_info=None
        self._download_quiet=bool(config.get('download_quiet',False))

    def _record(self,record,failed=False):
        # This method accepts only already-sanitized or internally generated data.
        with self._diagnostics_lock:
            self.diagnostics_path.parent.mkdir(parents=True,exist_ok=True)
            with self.diagnostics_path.open('a',encoding='utf8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        if failed: print('Kaggle CLI diagnostic: '+json.dumps(record,ensure_ascii=False),flush=True)

    def command(self,*args,timeout=None):
        return self._execute([self.cli,'datasets',*map(str,args)],'datasets '+str(args[0]),timeout)

    def cli_version(self): return self._execute([self.cli,'--version'],'--version',30)

    def _execute(self,argv,operation,timeout):
        start=time.monotonic()
        record=dict(timestamp_utc=utc(),operation=operation,
            argv=[sanitize_cli_output(a) for a in argv],stdout='',stderr='',returncode=None,timed_out=False)
        try:
            result=subprocess.run(argv,capture_output=True,timeout=self.timeout if timeout is None else timeout)
            record.update(stdout=sanitize_cli_output(result.stdout),stderr=sanitize_cli_output(result.stderr),
                          returncode=result.returncode)
        except subprocess.TimeoutExpired as exc:
            record.update(stdout=sanitize_cli_output(exc.stdout),stderr=sanitize_cli_output(exc.stderr),timed_out=True)
        except OSError as exc:
            record.update(stderr=sanitize_cli_output(str(exc)),launch_error=type(exc).__name__)
        record['elapsed_seconds']=round(time.monotonic()-start,6)
        failed=record['returncode']!=0
        self._record(record,failed=failed)
        if failed: raise KaggleCLIError(record) from None
        return record

    def file_names(self,deadline=None):
        """CLI 2.x CSV list; never infer presence from READY or upload exit code."""
        names=set();page_token=None
        for _ in range(10):
            args=['files',self.slug,'--csv','--page-size','200']
            if page_token: args+=['--page-token',page_token]
            record=self.command(*args,timeout=min(30,max(.001,deadline-time.monotonic())) if deadline else 30)
            text=record['stdout'];lines=text.splitlines()
            start=next((i for i,line in enumerate(lines) if line.lower().startswith(('name,','"name",'))),None)
            if start is None: raise RuntimeError('Kaggle files CSV has no name header; inspect CLI diagnostics')
            reader=csv.DictReader(io.StringIO('\n'.join(lines[start:])))
            for row in reader:
                name=row.get('name')
                if name and not name.startswith('Next Page Token'): names.add(name)
            match=re.search(r'Next Page Token\s*=\s*(\S+)',text)
            page_token=match[1] if match else None
            if not page_token: return names
        raise RuntimeError('Kaggle files pagination exceeded safety bound')

    def check_private(self):
        import requests
        token=os.environ.get('KAGGLE_API_TOKEN')
        auth=None if token else (os.environ.get('KAGGLE_USERNAME',''),os.environ.get('KAGGLE_KEY',''))
        headers={'Authorization':'Bearer '+token} if token else {}
        response=requests.get('https://www.kaggle.com/api/v1/datasets/view/'+self.slug,
                              auth=auth,headers=headers,timeout=30)
        if response.status_code!=200: raise RuntimeError('Cannot verify Kaggle Dataset privacy/access')
        data=response.json()
        if data.get('isPrivate') is not True:
            raise RuntimeError('Backup Dataset must already exist and be private (isPrivate=true)')
        self.last_dataset_info=dict(http_status=response.status_code,is_private=True,
            current_version_number=data.get('currentVersionNumber'),last_updated=data.get('lastUpdated'))
        return self.last_dataset_info

    def _backoff(self,attempt,deadline):
        delay=min(self.config.get('retry_initial_seconds',2)*2**min(attempt,20),
                  self.config.get('retry_max_seconds',15),max(0,deadline-time.monotonic()))
        if delay>0: time.sleep(delay)

    def fetch(self,destination,*,deadline=None,max_attempts=None):
        """Retry read-only downloads; never reuse artifacts from a failed attempt."""
        self.check_private();destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        limit=time.monotonic()+self.config.get('download_timeout_seconds',600)
        deadline=min(limit,deadline) if deadline is not None else limit
        attempts=max(1,int(max_attempts or self.config.get('download_attempts',3)))
        quiet=self._download_quiet
        for attempt in range(attempts):
            if time.monotonic()>=deadline: break
            try:
                with tempfile.TemporaryDirectory(dir=self.workspace,prefix='download_') as d:
                    args=['download',self.slug,'-p',d,'-o']+(['-q'] if quiet else [])
                    self.command(*args,timeout=min(self.timeout,max(.001,deadline-time.monotonic())))
                    archives=list(Path(d).glob('*.zip'))
                    if len(archives)!=1: raise ValueError('Download returned no unique Dataset ZIP despite exit code 0')
                    incoming=Path(d)/'incoming';incoming.mkdir();selected=[]
                    with zipfile.ZipFile(archives[0]) as z:
                        for name in z.namelist():
                            if '/' not in name and re.fullmatch(r'(training|probe)_[\w]+\.zip',name):
                                safe_relative(name)
                                if name in selected: raise ValueError('Duplicate logical ZIP in Dataset')
                                with z.open(name) as source,(incoming/name).open('wb') as dest: shutil.copyfileobj(source,dest)
                                selected.append(name)
                            elif '/' not in name and re.fullmatch(r'(training|probe)_[\w]+\.zip\.pcb64',name):
                                safe_relative(name);encoded=incoming/name
                                with z.open(name) as source,encoded.open('wb') as dest: shutil.copyfileobj(source,dest)
                                zip_name=name[:-len('.pcb64')]
                                if zip_name in selected: raise ValueError('Duplicate logical ZIP in Dataset')
                                decode_transport(encoded,incoming/zip_name);encoded.unlink()
                                selected.append(zip_name)
                    # No files become visible until the entire selected payload is read.
                    for name in selected: os.replace(incoming/name,destination/name)
                    return sorted((destination/n for n in selected if n.startswith('training_')),reverse=True)
            except (KaggleCLIError,zipfile.BadZipFile,EOFError,OSError,ValueError) as exc:
                self._record(dict(timestamp_utc=utc(),operation='download_attempt',attempt=attempt+1,
                    quiet=quiet,error_type=type(exc).__name__,detail=sanitize_cli_output(str(exc))),failed=True)
                if quiet:
                    # Safe read-only compatibility retry. This alone does not prove
                    # quiet mode caused the failure; the diagnostic compares both.
                    print('Quiet download failed; next attempt omits -q (output is still captured)',flush=True)
                    quiet=False
                    self._download_quiet=False
                if attempt+1<attempts: self._backoff(attempt,deadline)
        raise RuntimeError('Kaggle Dataset download not confirmed after bounded retries; inspect '+str(self.diagnostics_path)) from None

    def publish(self,bundle):
        self.check_private();bundle=Path(bundle);verify_bundle(bundle)
        expected_hash=sha256(bundle)
        upload_state=dict(timestamp_utc=utc(),dataset=self.slug,bundle=bundle.name,sha256=expected_hash,
                          upload_acknowledged=False,uploaded_snapshot_verified=False,dataset_metadata=self.last_dataset_info)
        expected_remote=transport_name(bundle.name)
        marker_name='publication_'+bundle.stem+'.json'
        upload_state.update(remote_object=expected_remote,publication_marker=marker_name,archive_transport='graphpcb-zip-base64-v1')
        state_path=self.workspace/'last_kaggle_publication.json'
        good=[]
        for p in sorted(self.cache.glob('training_*.zip'),reverse=True):
            try: verify_bundle(p);good.append(p)
            except Exception: continue
        with tempfile.TemporaryDirectory(dir=self.workspace,prefix='upload_') as d:
            stage=Path(d).resolve();originals=stage/'local_zip_audit';originals.mkdir()
            selected=list({p.name:p for p in [*good[:(2 if bundle.name.startswith('probe_') else 1)],bundle]}.values())
            resources=[]
            for path in selected:
                # Physical original ZIP remains auditable inside the exact staging
                # directory. -r skip deliberately excludes this diagnostic folder.
                local_zip=originals/path.name;shutil.copyfile(path,local_zip)
                encoded=stage/transport_name(path.name);header=encode_transport(local_zip,encoded)
                resources.append(dict(path=encoded.name,zip_name=path.name,size=encoded.stat().st_size,
                    sha256=sha256(encoded),zip_size=header['zip_size'],zip_sha256=header['zip_sha256']))
            atomic_json(stage/marker_name,dict(format='graphpcb-publication-v1',bundle=bundle.name,
                remote_object=expected_remote,zip_sha256=expected_hash,resources=resources))
            metadata=dict(id=self.slug,title='GraphPCB private checkpoint backups',licenses=[dict(name='other')],
                resources=[dict(path=r['path'],description='ASCII-armored exact ZIP checkpoint bytes') for r in resources]+
                          [dict(path=marker_name,description='Unique publication identity and expected ZIP hashes')])
            atomic_json(stage/'dataset-metadata.json',metadata)
            if not (originals/bundle.name).is_file() or sha256(originals/bundle.name)!=expected_hash:
                raise ValueError('Original snapshot ZIP missing/changed in staging directory')
            inventory=[dict(name=p.relative_to(stage).as_posix(),size=p.stat().st_size,sha256=sha256(p),
                            uploaded=p.parent==stage and p.name!='dataset-metadata.json')
                       for p in sorted(stage.rglob('*')) if p.is_file()]
            total=sum(r['size'] for r in inventory if r['uploaded'])
            audit=dict(timestamp_utc=utc(),operation='publication_staging',directory=sanitize_cli_output(str(stage)),
                dataset=self.slug,files=inventory,total_upload_bytes=total,dir_mode='skip',
                exclusions='metadata is configuration; local_zip_audit/ is intentionally local-only',
                remote_object=expected_remote,zip_sha256=expected_hash)
            upload_state['staging']=audit;atomic_json(state_path,upload_state)
            self._record(audit);print('Kaggle publication staging: '+json.dumps(audit),flush=True)
            if self.config.get('max_publication_bytes') and total>self.config['max_publication_bytes']:
                raise ValueError('Publication size '+str(total)+' exceeds configured max_publication_bytes; no upload attempted')
            try:
                # Capture output ourselves; avoid quiet-mode dependency for uploads.
                result=self.command('version','-p',str(stage),'-m','GraphPCB checkpoint '+bundle.stem,'-t','-r','skip')
                print('Kaggle publication CLI: '+json.dumps(result),flush=True)
                upload_state['upload_acknowledged']=True
            except KaggleCLIError:
                # A lost response/nonzero CLI exit can follow server acceptance.
                # Verify the uniquely named object before considering another upload.
                upload_state['upload_cli_failed']=True
        atomic_json(state_path,upload_state)
        # Processing is asynchronous; upload acknowledgement is NOT confirmation.
        deadline=time.monotonic()+self.config.get('verification_timeout_seconds',600)
        verification_attempts=max(1,int(self.config.get('verification_attempts',12)))
        for attempt in range(verification_attempts):
            if time.monotonic()>=deadline: break
            try:
                names=self.file_names(deadline)
                upload_state.update(remote_files=sorted(names),file_listing_attempt=attempt+1)
                atomic_json(state_path,upload_state)
                if expected_remote not in names:
                    if marker_name in names:
                        raise KagglePublicationMissingSnapshot(
                            'New publication marker exists but expected ZIP transport '+expected_remote+
                            ' is missing from datasets files; no checksum download attempted. Inspect staging/upload diagnostics.')
                    raise ValueError('New publication not yet visible in datasets files; no download attempted')
                with tempfile.TemporaryDirectory(dir=self.workspace,prefix='verify_') as d:
                    self.fetch(d,deadline=deadline,max_attempts=1);downloaded=Path(d)/bundle.name
                    if not downloaded.exists(): raise ValueError('Requested snapshot not yet in downloadable Dataset version')
                    if sha256(downloaded)!=expected_hash: raise ValueError('Downloaded snapshot SHA-256 mismatch')
                    verify_bundle(downloaded)
                    temp=self.cache/(bundle.name+'.verified.tmp');shutil.copyfile(downloaded,temp);temp.replace(self.cache/bundle.name)
                    if bundle.name.startswith('training_'):
                        for old in sorted(self.cache.glob('training_*.zip'),reverse=True)[2:]: old.unlink()
                    location='https://www.kaggle.com/datasets/'+self.slug+' [file '+expected_remote+'; original ZIP '+bundle.name+']'
                    upload_state.update(uploaded_snapshot_verified=True,verified_utc=utc(),location=location)
                    atomic_json(state_path,upload_state)
                    return location
            except KagglePublicationMissingSnapshot as exc:
                upload_state.update(last_error_type=type(exc).__name__,last_error=str(exc))
                atomic_json(state_path,upload_state)
                self._record(dict(timestamp_utc=utc(),operation='publication_missing_snapshot',detail=str(exc)),failed=True)
                raise
            except (RuntimeError,ValueError,zipfile.BadZipFile,EOFError,OSError) as exc:
                upload_state.update(verification_attempt=attempt+1,last_error_type=type(exc).__name__)
                atomic_json(state_path,upload_state)
                self._record(dict(timestamp_utc=utc(),operation='snapshot_readback',attempt=attempt+1,
                    bundle=bundle.name,error_type=type(exc).__name__,detail=sanitize_cli_output(str(exc))),failed=True)
                # Status is diagnostic evidence only; READY / rc=0 does not prove
                # this uniquely named new snapshot has propagated to downloads.
                if attempt==0 and time.monotonic()<deadline:
                    try: self.command('status',self.slug,timeout=min(30,max(.001,deadline-time.monotonic())))
                    except KaggleCLIError: pass
            if attempt+1<verification_attempts: self._backoff(attempt,deadline)
        if not upload_state.get('remote_files') or expected_remote not in upload_state.get('remote_files',[]):
            error=KagglePublicationMissingSnapshot('Expected ZIP transport '+expected_remote+
                ' never appeared in datasets files within bounded publication checks; previous versions retained; no wrong-version download attempted')
            upload_state.update(last_error_type=type(error).__name__,last_error=str(error))
            atomic_json(state_path,upload_state)
            self._record(dict(timestamp_utc=utc(),operation='publication_missing_snapshot',detail=str(error)),failed=True)
            raise error
        raise TimeoutError('Kaggle version not confirmed by complete download/checksum within verification time/attempt budget')


class S3Store:
    """Explicit private external S3 destination. Immutable keys + commit markers."""
    def __init__(self,config,workspace):
        import boto3
        from botocore.config import Config
        self.bucket=config['bucket'];self.prefix=config['prefix'].strip('/')+'/'
        if not self.bucket or self.prefix=='/': raise ValueError('Configure an explicit private bucket and dedicated prefix')
        endpoint=config.get('endpoint_url') or None
        if endpoint and not endpoint.startswith('https://'): raise ValueError('S3 endpoint must use HTTPS')
        self.client=boto3.client('s3',endpoint_url=endpoint,region_name=config.get('region') or None,
            config=Config(connect_timeout=15,read_timeout=120,retries=dict(max_attempts=3,mode='standard')))

    def publish(self,bundle):
        bundle=Path(bundle);key=self.prefix+bundle.name;digest=sha256(bundle)
        self.client.upload_file(str(bundle),self.bucket,key,ExtraArgs={'Metadata':{'sha256':digest}})
        # Full read-back verifies multipart uploads without trusting ETag as SHA256.
        h=hashlib.sha256();response=self.client.get_object(Bucket=self.bucket,Key=key)
        try:
            for block in response['Body'].iter_chunks(chunk_size=8*1024*1024): h.update(block)
        finally: response['Body'].close()
        if h.hexdigest()!=digest: raise ValueError('Remote S3 read-back hash mismatch')
        marker=dict(key=key,sha256=digest,size=bundle.stat().st_size,confirmed_utc=utc())
        marker_key=key+'.confirmed.json'
        self.client.put_object(Bucket=self.bucket,Key=marker_key,Body=json.dumps(marker).encode(),ContentType='application/json')
        response=self.client.get_object(Bucket=self.bucket,Key=marker_key)
        try: actual=json.loads(response['Body'].read())
        finally: response['Body'].close()
        if actual!=marker: raise ValueError('S3 confirmation marker read-back failed')
        return f's3://{self.bucket}/{key}'

    def fetch(self,destination):
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);keys=[]
        for page in self.client.get_paginator('list_objects_v2').paginate(Bucket=self.bucket,Prefix=self.prefix+'training_'):
            keys.extend(o['Key'] for o in page.get('Contents',[]) if o['Key'].endswith('.confirmed.json'))
        found=[]
        for key in sorted(keys,reverse=True):
            if len(found)>=2: break
            try:
                response=self.client.get_object(Bucket=self.bucket,Key=key)
                try: marker=json.loads(response['Body'].read())
                finally: response['Body'].close()
                if marker['key']+'.confirmed.json'!=key: raise ValueError('Marker key mismatch')
                name=marker['key'].split('/')[-1];safe_relative(name)
                target=destination/name;self.client.download_file(self.bucket,marker['key'],str(target))
                if target.stat().st_size!=marker['size'] or sha256(target)!=marker['sha256']: raise ValueError('S3 downloaded hash mismatch')
                verify_bundle(target);found.append(target)
            except Exception as exc:
                print('WARNING: rejected S3 backup candidate:',type(exc).__name__,flush=True)
        if keys and not found: raise RuntimeError('No valid confirmed S3 backup is recoverable')
        return found


def make_store(config,workspace):
    Path(workspace).mkdir(parents=True,exist_ok=True)
    provider=config.get('provider')
    if provider=='kaggle': return KaggleStore(config,workspace)
    if provider=='s3': return S3Store(config,workspace)
    raise ValueError('Configure provider=kaggle or provider=s3; no local persistence fallback')


def recover(store,workspace,run,attached=None,allow_new=False):
    """Never start over merely because the network is unavailable or a ZIP is bad."""
    workspace=Path(workspace);workspace.mkdir(parents=True,exist_ok=True)
    candidates=sorted(Path(attached).glob('training_*.zip'),reverse=True) if attached else store.fetch(workspace)
    if attached:
        for encoded in sorted(Path(attached).glob('training_*.zip.pcb64'),reverse=True):
            target=workspace/encoded.name[:-len('.pcb64')]
            try: decode_transport(encoded,target);candidates.append(target)
            except Exception:
                # Keep an invalid candidate so allow_new cannot hide failed recovery.
                candidates.append(encoded)
    valid=[]
    for p in candidates:
        try:
            m=verify_bundle(p,Path(run).name)
            if m['kind']=='training': valid.append((m['created_utc'],p))
        except Exception as exc: print('WARNING: rejected backup:',p.name,type(exc).__name__,flush=True)
    if not valid:
        if candidates or not allow_new: raise RuntimeError('No valid backup found; refusing to silently restart. Check run name/input or explicitly allow a new run.')
        print('Explicitly allowed new run: no remote training snapshot exists',flush=True);return None
    path=max(valid)[1];manifest=restore_bundle(path,run)
    # Seed Kaggle retention cache with valid remote predecessors after a fresh session.
    if isinstance(store,KaggleStore):
        for _,p in valid[:2]: shutil.copyfile(p,store.cache/p.name)
    atomic_json(Path(run)/'recovery_verified.json',dict(restored_utc=utc(),bundle=path.name,manifest=manifest))
    location=('https://www.kaggle.com/datasets/'+store.slug if isinstance(store,KaggleStore)
              else ('s3://'+store.bucket+'/'+store.prefix if isinstance(store,S3Store) else 'TEST STORAGE'))
    atomic_json(Path(run)/'persistent_backup_status.json',dict(last_success_utc=utc(),
        last_success_location=location+' [restored '+path.name+']',last_error=None))
    return manifest


def preflight(store,workspace):
    """A real tiny upload + read-back. Never implied by mocked tests."""
    workspace=Path(workspace);workspace.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace,prefix='probe_run_') as d:
        run=Path(d)/'storage-probe';run.mkdir();atomic_json(run/'run_settings.json',dict(nonce=uuid.uuid4().hex))
        bundle=build_bundle(run,workspace,kind='probe')
        try: location=store.publish(bundle)
        finally: bundle.unlink(missing_ok=True)
    print('LIVE persistent upload/read-back preflight verified:',location,flush=True)
    return dict(verified_utc=utc(),location=location)


class BackupManager:
    def __init__(self,run,store,workspace,interval_seconds=3600,retry_seconds=60):
        self.run=Path(run);self.store=store;self.workspace=Path(workspace)
        self.interval=interval_seconds;self.retry=retry_seconds
        self.lock=threading.RLock();self.wake=threading.Event();self.stop_event=threading.Event()
        self.status=dict(last_success_utc=None,last_success_location=None,last_error=None)
        self.status_lock=threading.Lock();self.thread=None
        self.status_path=self.run/'persistent_backup_status.json'
        if self.status_path.exists(): self.status.update(json.loads(self.status_path.read_text()))

    def backup_once(self):
        bundle=None
        try:
            print('Persistent backup started (last committed epoch)',flush=True)
            bundle=build_bundle(self.run,self.workspace,self.lock)
            location=self.store.publish(bundle);manifest=verify_bundle(bundle,self.run.name)
            with self.status_lock:
                self.status.update(last_success_utc=utc(),last_success_location=location,last_error=None)
                atomic_json(self.status_path,self.status)
            for row in manifest['checkpoints']:
                print(f'Persistent backup verified — experiment: {row["trial"]}, completed epoch: {row["completed_epoch"]}',flush=True)
            return True
        except Exception as exc:
            # Exception messages from HTTP clients can contain signed URLs/tokens.
            with self.status_lock:
                self.status['last_error']=type(exc).__name__
                atomic_json(self.status_path,self.status)
            print('WARNING: persistent backup NOT confirmed ('+type(exc).__name__+'); training can continue; automatic retry scheduled',flush=True)
            return False
        finally:
            if bundle is not None: bundle.unlink(missing_ok=True)

    def _loop(self):
        next_hour=time.monotonic()+self.interval;retry_at=float('inf')
        while True:
            requested=self.wake.wait(max(0,min(next_hour,retry_at)-time.monotonic()));self.wake.clear()
            if self.stop_event.is_set(): return
            now=time.monotonic()
            if not requested and now<min(next_hour,retry_at): continue
            while next_hour<=now: next_hour+=self.interval
            print('Hourly/requested persistent backup started',flush=True)
            retry_at=float('inf') if self.backup_once() else time.monotonic()+self.retry

    def start(self):
        self.thread=threading.Thread(target=self._loop,name='persistent-checkpoint-backup',daemon=True)
        self.thread.start();return self

    def request(self): self.wake.set()

    def finish(self,timeout_seconds=900):
        # Serialize final upload behind an in-flight hourly upload. A hung/slow
        # provider cannot block shutdown forever; old confirmed versions survive.
        self.stop_event.set();self.wake.set()
        deadline=time.monotonic()+timeout_seconds
        if self.thread: self.thread.join(max(0,deadline-time.monotonic()))
        if self.thread and self.thread.is_alive():
            print('WARNING: final persistent backup could not complete before shutdown deadline',flush=True)
            return False
        result=[]
        def final_attempts():
            for attempt in range(3):
                if time.monotonic()>=deadline: break
                if self.backup_once(): result.append(True);return
                time.sleep(min(2**attempt,max(0,deadline-time.monotonic())))
            result.append(False)
        final=threading.Thread(target=final_attempts,daemon=True)
        final.start();final.join(max(0,deadline-time.monotonic()))
        success=bool(result and result[0])
        if not success: print('WARNING: FINAL PERSISTENT BACKUP FAILED OR TIMED OUT; only previously confirmed backups are recoverable',flush=True)
        return success


class SessionBudget:
    def __init__(self,hours,buffer_seconds=900):
        self.start=time.monotonic();self.deadline=self.start+hours*3600 if hours else float('inf')
        self.buffer=buffer_seconds
        if hours and hours*3600<=buffer_seconds: raise ValueError('SESSION_HOURS must exceed SHUTDOWN_BUFFER_SECONDS')
    def should_stop(self): return time.monotonic()>=self.deadline-self.buffer
    def remaining(self,maximum=900): return max(0,min(maximum,self.deadline-time.monotonic()))
