"""Direct power execution keeps provisional results distinct from validation."""
import json

import numpy as np
import pytest
from scipy.io import loadmat

from simulation.finite_strip import prepare_strip_case, run_strip_case
from Compute.finite_strip.analyze import export_kwave_input
from verification.test_strip_30mm import _export_fixture


def test_power_progress_preserves_accepted_numerical_result(tmp_path):
    config = prepare_strip_case(dict(name='progress', kind='power', dx_m=1e-5,
        cycles=.02, sponge_width_m=0., sponge_rate=0., stop_when_stable=False))
    observed = []
    run_strip_case(config, tmp_path/'baseline')
    run_strip_case(config, tmp_path/'observed', progress_callback=observed.append)
    assert observed and not observed[-1]['run_completed']
    with np.load(tmp_path/'baseline/trajectory.npz') as first, np.load(tmp_path/'observed/trajectory.npz') as second:
        for key in ('film_temperature_K', 'pressure_perturbation_Pa', 'audit'):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(observed[-1]['film_temperature_K'], second['film_temperature_K'])
    assert json.loads((tmp_path/'observed/summary.json').read_text())['validation_passed'] is None


def test_unvalidated_export_preserves_failed_checks_and_source_flag(tmp_path):
    result, arrays = _export_fixture()
    result.update(source_validation_passed=False, checks=dict(run_completed=True,
        fitted_signals_finite_and_conditioned=True, geometry=True, boundary_prerequisite=False))
    path = tmp_path/'direct.mat'
    export_kwave_input(path, result, arrays, toolbox_path=tmp_path, unvalidated=True)
    saved = loadmat(path, squeeze_me=True)
    assert saved['source_validation_passed'] == 0
    assert saved['diagnostic_mode'] == saved['source_validation_bypassed'] == 1
    assert not json.loads(str(saved['source_analysis_json']))['checks']['boundary_prerequisite']


@pytest.mark.parametrize('failed', ['run_completed', 'fitted_signals_finite_and_conditioned', 'geometry'])
def test_unvalidated_export_still_requires_usable_data(tmp_path, failed):
    result, arrays = _export_fixture()
    result.update(source_validation_passed=False, checks=dict(run_completed=True,
        fitted_signals_finite_and_conditioned=True, geometry=True))
    result['checks'][failed] = False
    with pytest.raises(ValueError, match='usable'):
        export_kwave_input(tmp_path/'bad.mat', result, arrays, toolbox_path=tmp_path, unvalidated=True)


def test_direct_route_waits_for_memory_then_starts_only_power(tmp_path, monkeypatch):
    from Compute.finite_strip import direct
    config=dict(numerics={'collision_backend':'cupy'},strip={'kind':'power','steps':12})
    monkeypatch.setattr(direct,'load_cases',lambda _: ({},{'power_r2':config}))
    responses=iter(['GPU-one, 2000\n','GPU-one, 24000\n'])
    commands=[];waits=[]
    def query(command,**kwargs):
        commands.append(command)
        return next(responses)
    monkeypatch.setattr(direct.subprocess,'check_output',query)
    monkeypatch.setattr(direct.time,'sleep',waits.append)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','previous')
    class StopAfterDispatch(Exception):
        pass
    def run(actual,output,*,progress_callback):
        assert actual['execution']['boundary_validation']=='skipped_by_user_request'
        assert output.name=='power_r2'
        assert direct.os.environ['CUDA_VISIBLE_DEVICES']=='GPU-one'
        progress_callback(dict(accepted_steps=1,physical_time_s=1e-8,elapsed_s=.5,
            frequency_Hz=20000.,film_temperature_K=np.array([[300.01]]),
            pressure_perturbation_Pa=np.array([[.002]]),run_completed=False))
        raise StopAfterDispatch()
    monkeypatch.setattr(direct,'run_strip_case',run)
    matlab=tmp_path/'matlab';matlab.touch()
    (tmp_path/'kspaceFirstOrder2D.m').touch()
    with pytest.raises(StopAfterDispatch):
        direct.main(['--study',str(tmp_path/'study.yaml'),'--batch','direct',
            '--matlab',str(matlab),'--toolbox',str(tmp_path)])
    assert waits==[30] and all(command[0]=='nvidia-smi' for command in commands)
    progress=json.loads((tmp_path/'direct/_work/progress.json').read_text())
    assert progress['validation_passed'] is None and not progress['run_completed']
    status=json.loads((tmp_path/'direct/_work/run_status.json').read_text())
    assert status['stage']=='failed' and status['error']['type']=='StopAfterDispatch'
