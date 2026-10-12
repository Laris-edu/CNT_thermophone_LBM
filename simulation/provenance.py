"""Read-only Git and source-hash records; never archive or copy source files."""
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess

SOURCE_DIRS=('core','boundary','coupling','phase3_interfaces','farfield','postproc',
             'reference','simulation','scripts','configs')


def capture_numba_runtime(cell_count):
    """Record effective dispatch; environment settings are not source hashes."""
    import llvmlite
    import numba
    import numpy as np
    from core.collision_numba import PARALLEL_MIN_CELLS, _parallel_field

    # get_num_threads initializes the selected threading layer, including on
    # a serial-sized case. It does not claim that its kernels ran in parallel.
    threads = numba.get_num_threads()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    available = (len(affinity) if affinity is not None else
                 (getattr(os, 'process_cpu_count', os.cpu_count)() or 1))
    return dict(cell_count=int(cell_count), parallel_min_cells=PARALLEL_MIN_CELLS,
        parallel_selected=bool(_parallel_field(cell_count)), numba_threads=threads,
        numba_thread_limit=numba.config.NUMBA_NUM_THREADS,
        threading_layer=numba.threading_layer(), cpu_available=available,
        cpu_logical_count=os.cpu_count(), cpu_affinity=affinity,
        platform=platform.platform(), machine=platform.machine(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      numba=numba.__version__, llvmlite=llvmlite.__version__),
        environment={name:os.environ.get(name) for name in (
            'NUMBA_NUM_THREADS', 'NUMBA_THREADING_LAYER', 'CNT_LBM_PARALLEL_MIN_CELLS',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
            'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'NUMEXPR_NUM_THREADS')})


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
