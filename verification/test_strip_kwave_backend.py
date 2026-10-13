"""Optional real MATLAB backend smoke checks; the coarse fixture tests no physics."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from verification.test_strip_handoff import ADAPTER, MATLAB, ROOT, TOOLBOX, _input


def _run(tmp_path, backend):
    if os.environ.get('RUN_KWAVE_GATE_TESTS') != '1':
        pytest.skip('set RUN_KWAVE_GATE_TESTS=1 for the MATLAB backend smoke checks')
    if not MATLAB.is_file() or not TOOLBOX.is_dir():
        pytest.skip('MATLAB and k-Wave are required')
    config = _input(.002)
    config.update(domain_half_width_m=.012, domain_half_height_m=.012,
                  observer_x_m=np.array([0.]), observer_y_m=np.array([.010]),
                  evaluation_cycles=3.25, ramp_cycles=1., record_cycles=1.,
                  cfl=.3, pml_size=4, kwave_backend_requested=backend)
    incoming, outgoing = tmp_path/'in.mat', tmp_path/'out.mat'
    error = tmp_path/'error.mat'
    savemat(incoming, config)
    quote = lambda p: "'"+Path(p).as_posix().replace("'", "''")+"'"
    command = (f"addpath({quote(ADAPTER.parent)});failure='';"
               f"try,kwave_handoff({quote(incoming)},{quote(outgoing)});"
               f"catch exception,failure=exception.message;end;save({quote(error)},'failure','-v7')")
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    completed = subprocess.run([str(MATLAB), '-singleCompThread', '-batch', command],
                               cwd=ROOT, capture_output=True, timeout=90, **options)
    assert completed.returncode == 0, (completed.stdout+completed.stderr).decode(errors='replace')
    failure = loadmat(error, squeeze_me=True)['failure']
    return outgoing, str(failure) if np.size(failure) else ''


def test_matlab_cpu_records_effective_backend_and_cpu_arrays(tmp_path):
    path, error = _run(tmp_path, 'cpu')
    assert not error
    result = loadmat(path, squeeze_me=True)
    assert result['kwave_backend_requested'] == result['kwave_backend_effective'] == 'cpu'
    assert result['data_cast'] == 'double'
    assert result['observer_pressure_Pa'].dtype == np.float64
    assert np.isfinite(result['observer_pressure_Pa']).all()


def test_matlab_rejects_unknown_backend_without_output(tmp_path):
    path, error = _run(tmp_path, 'silent-fallback')
    assert 'backend' in error.lower()
    assert not path.exists()


def test_matlab_gpu_preserves_double_cpu_pressure(tmp_path):
    if os.environ.get('RUN_KWAVE_GPU_TESTS') != '1':
        pytest.skip('set RUN_KWAVE_GPU_TESTS=1 on a supported MATLAB GPU host')
    cpu_dir, gpu_dir = tmp_path/'cpu', tmp_path/'gpu'
    cpu_dir.mkdir(); gpu_dir.mkdir()
    cpu_path, error = _run(cpu_dir, 'cpu')
    assert not error
    gpu_path, error = _run(gpu_dir, 'matlab-gpu')
    assert not error
    cpu, gpu = (loadmat(path, squeeze_me=True) for path in (cpu_path, gpu_path))
    assert gpu['kwave_backend_effective'] == 'matlab-gpu'
    assert gpu['data_cast'] == 'gpuArray-double'
    assert gpu['observer_pressure_Pa'].dtype == np.float64
    np.testing.assert_allclose(gpu['observer_pressure_Pa'], cpu['observer_pressure_Pa'], rtol=1e-10, atol=1e-12)
