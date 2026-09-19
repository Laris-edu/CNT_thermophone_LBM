"""Record a fixed Git revision, or share one snapshot of the actual source bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

SOURCE_DIRS=('core','boundary','coupling','phase3_interfaces','farfield','postproc',
             'reference','simulation','scripts','configs')


def source_files(root):
    root=Path(root)
    return sorted({p for directory in SOURCE_DIRS for p in (root/directory).rglob('*')
        if p.is_file() and p.suffix in ('.py','.yaml')} |
        {root/name for name in ('config_io.py','requirements.txt','requirements-lock.txt','pyproject.toml')
         if (root/name).is_file()})


def source_snapshot(root, sources, output):
    """Compatibility API: return archive path and per-file source hashes."""
    root=Path(root); output=Path(output)
    content={p.relative_to(root).as_posix():p.read_bytes() for p in sorted(set(sources))}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in content.items()}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode('utf-8')).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    path=output/f'{version}.zip'
    if not path.exists():
        with ZipFile(path,'x',ZIP_DEFLATED) as archive:
            for name,data in content.items():
                archive.writestr(name,data)
    with ZipFile(path) as archive:
        if set(archive.namelist())!=set(content) or any(
            hashlib.sha256(archive.read(name)).hexdigest()!=digest for name,digest in hashes.items()):
            raise ValueError(f'source archive content mismatch: {path}')
    return path,hashes


def capture_source(root, cache):
    root=Path(root).resolve()
    files=source_files(root)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode('utf-8')).hexdigest()
    commit=None; clean=False
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,
                              encoding='utf-8',errors='replace',check=True).stdout.strip()
    try:
        if Path(git('rev-parse','--show-toplevel')).resolve()==root:
            commit=git('rev-parse','HEAD')
            tracked=set(git('ls-files').splitlines())
            # Include deleted runtime files: they no longer appear in the current file list.
            clean=set(hashes).issubset(tracked) and not git('diff','--name-only','HEAD','--',
                *SOURCE_DIRS,'config_io.py','requirements.txt','requirements-lock.txt','pyproject.toml')
    except (OSError,subprocess.CalledProcessError):
        pass
    snapshot=None
    if not clean:
        snapshot,archived_hashes=source_snapshot(root,files,cache)
        if archived_hashes!=hashes:
            raise RuntimeError('source changed while preparing the run')
    return dict(source_id=source_id,git_commit=commit,source_clean=clean,
                source_sha256=hashes,snapshot=str(snapshot.resolve()) if snapshot else None)
