"""Cloud dispatch must be explicit and its effective settings reproducible."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def probe(threshold=None, threads=2):
    env = os.environ.copy()
    env.pop('CNT_LBM_PARALLEL_MIN_CELLS', None)
    if threshold is not None:
        env['CNT_LBM_PARALLEL_MIN_CELLS'] = threshold
    env['NUMBA_NUM_THREADS'] = str(threads)
    return subprocess.run([sys.executable, '-c',
        'import json; from core.collision_numba import _parallel_field; '
        'from simulation.provenance import capture_numba_runtime; '
        'print(json.dumps(capture_numba_runtime(234396)))'],
        cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize('threshold,threads,parallel', [
    (None, 2, False), ('131072', 2, True), ('131072', 1, False),
])
def test_runtime_records_the_dispatch_used_by_this_process(threshold, threads, parallel):
    result = probe(threshold, threads)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record['parallel_selected'] is parallel
    assert record['numba_threads'] == threads
    assert record['numba_thread_limit'] == threads
    assert record['parallel_min_cells'] == (262144 if threshold is None else 131072)
    assert record['cell_count'] == 234396
    assert record['threading_layer'] in ('omp', 'tbb', 'workqueue')
    assert record['versions']['numba']
    assert record['versions']['numpy']
    assert record['cpu_available'] >= 1


@pytest.mark.parametrize('threshold', ['', '0', '-2', '1.5', 'typo'])
def test_invalid_parallel_threshold_fails_before_a_solver_can_start(threshold):
    result = probe(threshold)
    assert result.returncode != 0
    assert 'CNT_LBM_PARALLEL_MIN_CELLS must be a positive integer' in result.stderr
