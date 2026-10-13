"""GPU observation/output boundaries and truthful runtime provenance."""
import json

import numpy as np
import pytest

from simulation.finite_strip import _sample_bilinear, prepare_strip_case, run_strip_case
from verification.test_acoustic_pml_gpu import cp


def pulse_parameters(**extra):
    return dict(name='lifecycle', kind='pulse', dx_m=1e-5,
        width_m=.00032, height_m=.00018, duration_s=2e-7,
        pulse_sigma_m=2e-5, probe_radius_m=5e-5,
        sponge_width_m=0, sponge_rate=0, **extra)


def test_invalid_strip_backend_is_rejected_before_execution():
    with pytest.raises(ValueError, match='collision_backend'):
        prepare_strip_case(pulse_parameters(collision_backend='cudaa'))


def test_source_hash_tracks_optional_gpu_dependency_changes(tmp_path):
    from simulation.provenance import capture_source

    dependency = tmp_path/'requirements-gpu.txt'
    dependency.write_text('cupy-cuda12x==14.2.0\n', encoding='utf-8')
    initial = capture_source(tmp_path)
    assert 'requirements-gpu.txt' in initial['source_sha256']
    dependency.write_text('cupy-cuda12x==14.2.1\n', encoding='utf-8')
    changed = capture_source(tmp_path)
    assert changed['source_id'] != initial['source_id']


def test_explicit_numpy_strip_records_numpy_runtime(tmp_path):
    config = prepare_strip_case(pulse_parameters(collision_backend='numpy'))
    config['strip']['steps'] = 1
    output = tmp_path/'numpy_run'
    run_strip_case(config, output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['collision_backend'] == 'numpy'
    assert manifest['runtime']['versions']['numpy'] == np.__version__
    assert 'numba_threads' not in manifest['runtime']


def test_gpu_bilinear_sampling_returns_only_host_observations(cp):
    x = np.array([-2., -.8, .5, 2.])
    y = np.array([.1, .6, 1.5])
    xx, yy = np.meshgrid(x, y)
    values = 2+3*xx-4*yy+.7*xx*yy
    px = np.array([-2., 2., -.3, .5, 1.1])
    py = np.array([.1, 1.5, .9, .6, .2])
    result = _sample_bilinear(cp.asarray(values), x, y, px, py)
    assert isinstance(result, np.ndarray) and result.shape == px.shape
    np.testing.assert_allclose(result, 2+3*px-4*py+.7*px*py, rtol=2e-15, atol=2e-15)
    with pytest.raises(ValueError):
        _sample_bilinear(cp.asarray(values), x, y, np.array([2.01]), np.array([.2]))


def test_cupy_runtime_records_actual_device_versions_and_memory(cp):
    from simulation.provenance import capture_cupy_runtime

    result = capture_cupy_runtime(576)
    assert result['cell_count'] == 576
    assert result['device_id'] == cp.cuda.Device().id
    assert result['device_name']
    assert result['precision'] == 'float64'
    assert result['versions']['cupy'] == cp.__version__
    assert result['cuda_runtime_version'] == cp.cuda.runtime.runtimeGetVersion()
    assert result['cuda_driver_version'] == cp.cuda.runtime.driverGetVersion()
    assert 0 < result['free_memory_bytes'] <= result['total_memory_bytes']
    assert 'numba_threads' not in result
    json.dumps(result)


def test_failed_gpu_run_saves_last_accepted_fields_and_failure_status(cp, tmp_path, monkeypatch):
    import simulation.finite_strip as strip

    config = prepare_strip_case(pulse_parameters(collision_backend='cupy'))
    accepted = {}

    def fail_after_step(solver, run):
        solver.step_temperature(300.01)
        state = solver.get_macro()
        accepted.update(pressure=cp.asnumpy(state.p*solver.mapping.pressure_scale),
            temperature=cp.asnumpy(state.theta*solver.mapping.temperature_scale),
            velocity=cp.asnumpy(state.u*solver.mapping.velocity_scale))
        raise RuntimeError('after accepted GPU step')

    monkeypatch.setattr(strip, 'collect_strip', fail_after_step)
    output = tmp_path/'failed_gpu'
    with pytest.raises(RuntimeError, match='after accepted GPU step'):
        run_strip_case(config, output)
    manifest = json.loads((output/'manifest.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    assert manifest['status'] == summary['status'] == 'failed'
    assert manifest['runtime']['device_id'] == cp.cuda.Device().id
    assert manifest['runtime_started']['precision'] == 'float64'
    assert summary['validation_passed'] is None
    with np.load(output/'trajectory.npz') as data:
        assert not data['run_completed'] and data['accepted_steps'] == 1
        for key, expected in (('last_pressure_Pa', accepted['pressure']),
                              ('last_temperature_K', accepted['temperature']),
                              ('last_velocity_m_s', accepted['velocity'])):
            np.testing.assert_array_equal(data[key], expected)


@pytest.mark.parametrize('recovery_failure,runtime_failure', [
    ('macro', False), ('download', False), (None, True),
    ('macro', True), ('download', True),
])
def test_device_error_while_reporting_failure_keeps_original_and_failed_metadata(
        tmp_path, monkeypatch, recovery_failure, runtime_failure):
    """Inject unavailable-device errors only at recovery/runtime boundaries.

    A real accepted step precedes the original error. CPU execution makes the
    error-reporting regression runnable even on machines without CUDA.
    """
    import simulation.finite_strip as strip
    import simulation.provenance as provenance

    config = prepare_strip_case(pulse_parameters(collision_backend='numpy'))
    original = RuntimeError('primary GPU stepping failure')
    capture = provenance.capture_numpy_runtime
    started_runtime = []

    def capture_with_device_failure(cells):
        if started_runtime and runtime_failure:
            raise OSError('device unavailable during runtime query')
        value = capture(cells)
        started_runtime.append(value)
        return value

    def unavailable_macro():
        raise OSError('device unavailable during state recovery')

    def unavailable_download(value):
        raise MemoryError('device download failed')

    def fail_after_step(solver, run):
        solver.step_temperature(300.01)
        if recovery_failure == 'macro':
            monkeypatch.setattr(solver, 'get_macro', unavailable_macro)
        elif recovery_failure == 'download':
            monkeypatch.setattr(strip, 'to_numpy', unavailable_download)
        raise original

    monkeypatch.setattr(provenance, 'capture_numpy_runtime', capture_with_device_failure)
    monkeypatch.setattr(strip, 'collect_strip', fail_after_step)
    output = tmp_path/'failed_device'
    with pytest.raises(RuntimeError) as raised:
        run_strip_case(config, output)
    assert raised.value is original
    manifest = json.loads((output/'manifest.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    assert manifest['status'] == summary['status'] == 'failed'
    assert manifest['error'] == summary['error'] == dict(
        type='RuntimeError', message='primary GPU stepping failure')
    assert summary['validation_passed'] is None
    assert manifest['finished_utc'] is not None
    assert manifest['runtime_started'] == started_runtime[0]
    assert manifest['runtime'] == started_runtime[0]
    stages = [error['stage'] for error in manifest['secondary_errors']]
    expected_stages = (['state_recovery'] if recovery_failure else [])
    expected_stages += ['runtime_capture'] if runtime_failure else []
    assert stages == expected_stages
    assert summary['secondary_errors'] == manifest['secondary_errors']
    assert all(error['type'] and error['message'] for error in manifest['secondary_errors'])
    with np.load(output/'trajectory.npz') as data:
        assert not data['run_completed']
        if recovery_failure:
            assert data['t_s'].size == 0 and 'last_pressure_Pa' not in data
        else:
            assert data['accepted_steps'] == 1
            assert np.isfinite(data['last_pressure_Pa']).all()
