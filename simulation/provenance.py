"""Read-only Git and source-hash records; never archive or copy source files."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_DIRS=('core','boundary','coupling','phase3_interfaces','farfield','postproc',
             'reference','simulation','scripts','configs')


def source_files(root):
    root=Path(root)
    return sorted({p for directory in SOURCE_DIRS for p in (root/directory).rglob('*')
        if p.is_file() and p.suffix in ('.py','.yaml')} |
        {root/name for name in ('config_io.py','requirements.txt','requirements-lock.txt','pyproject.toml')
         if (root/name).is_file()})


def capture_source(root, cache=None):
    """Record the commit, repository status and public runtime source hashes.

    git_dirty covers the whole repository (including case scripts); source_clean
    only describes the public runtime source set. Unknown Git status is None,
    not a claim of cleanliness. The former cache argument is accepted but ignored
    so existing case scripts no longer create archives without needing edits.
    """
    root=Path(root).resolve()
    files=source_files(root)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode('utf-8')).hexdigest()
    commit=None; clean=False; dirty=None
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,
                              encoding='utf-8',errors='replace',check=True).stdout.strip()
    try:
        if Path(git('rev-parse','--show-toplevel')).resolve()==root:
            dirty=bool(git('status','--porcelain','--untracked-files=normal'))
            commit=git('rev-parse','HEAD')
            tracked=set(git('ls-files','-z').split('\0'))
            # Include deleted runtime files: they no longer appear in the current file list.
            clean=set(hashes).issubset(tracked) and not git('diff','--name-only','HEAD','--',
                *SOURCE_DIRS,'config_io.py','requirements.txt','requirements-lock.txt','pyproject.toml')
    except (OSError,subprocess.CalledProcessError):
        pass
    return dict(source_id=source_id,git_commit=commit,git_dirty=dirty,
                source_clean=clean,source_sha256=hashes)
