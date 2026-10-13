"""Numerical propagation gate; fabricated arrays test decisions, not physics."""
import numpy as np
import pytest

from Compute.finite_strip.chain import validate_propagation
from farfield.strip_surface import upper_rectangle,mirror_strip_surface
from scipy.special import hankel2


def fixture():
    frequency,rho,c=20000.,1.16,347.
    upper=upper_rectangle(.0007,.0002,5e-6)
    k=2*np.pi*frequency/c
    r=np.hypot(upper['x_m'],upper['y_m'])
    p=hankel2(0,k*r)*np.exp(.4j)
    ur=-1j*hankel2(1,k*r)*np.exp(.4j)/(rho*c)
    closed=mirror_strip_surface(upper,p,ur*upper['x_m']/r,ur*upper['y_m']/r)
    ox,oy=np.array([0.,0.,0.]),np.array([.002,.045,0.])
    hat=hankel2(0,k*oy[:2])*np.exp(.4j)
    hat=np.r_[hat,0j]
    t=np.arange(1,129)/(64*frequency)
    source=dict(**closed,frequency_Hz=frequency,rho0_kg_m3=rho,c0_m_s=c,
        source_validation_passed=1,observer_x_m=ox,observer_y_m=oy,
        expected_grid_observer_x_m=ox,expected_grid_observer_y_m=oy,
        observer_role=np.array(['nearfield','farfield_45mm','interior_cancel']))
    result=dict(observer_x_m=ox,observer_y_m=oy,observer_p_hat_Pa=hat,
                time_s=t,observer_pressure_Pa=np.real(hat[:,None]*np.exp(2j*np.pi*frequency*t)))
    return source,result


def test_propagation_gate_uses_actual_coordinates_and_absolute_phase():
    source,result=fixture()
    check=validate_propagation(source,result)
    assert check['propagation_validation_passed']
    result['observer_p_hat_Pa']=result['observer_p_hat_Pa']*np.exp(.03j)
    check=validate_propagation(source,result)
    assert not check['propagation_validation_passed']
    assert check['max_phase_error_deg']>1.


def test_interior_leakage_is_a_separate_gate():
    source,result=fixture()
    result['observer_p_hat_Pa'][-1]=.02*max(abs(result['observer_p_hat_Pa']))
    check=validate_propagation(source,result)
    assert check['max_amplitude_error_relative']<.01
    assert check['interior_leakage_relative']>.01
    assert not check['propagation_validation_passed']


def test_coordinate_mismatch_is_not_compared_as_if_same_position():
    source,result=fixture()
    result['observer_y_m']=result['observer_y_m']+.00005
    with pytest.raises(ValueError,match='coordinates'):
        validate_propagation(source,result)


def test_saved_time_series_must_match_reported_complex_pressure():
    source,result=fixture()
    result['observer_pressure_Pa']*=2
    check=validate_propagation(source,result)
    assert not check['propagation_validation_passed']


@pytest.mark.parametrize('damage',['truncated','gap','duplicate'])
def test_propagation_requires_two_complete_regularly_sampled_periods(damage):
    source,result=fixture()
    if damage=='truncated':
        selection=np.arange(32,len(result['time_s']))
    elif damage=='gap':
        selection=np.delete(np.arange(len(result['time_s'])),40)
    else:
        selection=np.r_[np.arange(40),39,np.arange(40,len(result['time_s']))]
    result['time_s']=result['time_s'][selection]
    result['observer_pressure_Pa']=result['observer_pressure_Pa'][:,selection]
    with pytest.raises(ValueError,match='time'):
        validate_propagation(source,result)


def _saved_stage(tmp_path, monkeypatch):
    import json
    import yaml
    from Compute.finite_strip import chain
    root = tmp_path/'downloaded'
    raw = root/'raw/power_r2'
    raw.mkdir(parents=True)
    config = dict(case=dict(name='power_r2'), strip=dict(kind='power', height_m=.0002))
    (raw/'effective.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (raw/'manifest.json').write_text(json.dumps(dict(status='completed', source_id='original_cloud_runtime',
                                                    source_clean=True, git_commit='cloud_commit')))
    (raw/'summary.json').write_text(json.dumps(dict(status='completed')))
    np.savez(raw/'trajectory.npz', t_s=np.array([0.]))
    matlab = tmp_path/'matlab.exe'; matlab.touch()
    toolbox = tmp_path/'local_kwave'; toolbox.mkdir()
    (toolbox/'kspaceFirstOrder2D.m').touch()
    monkeypatch.setattr(chain, 'load_cases', lambda *a: pytest.fail('read current study'))
    monkeypatch.setattr(chain, 'run_strip_case', lambda *a: pytest.fail('restarted LBM'))
    def evidence(directory, actual, **kwargs):
        assert directory == root and actual == config
        assert kwargs['source_id'] == 'original_cloud_runtime'
        return dict(boundary_validation_passed=True)
    monkeypatch.setattr(chain, 'validate_boundary_evidence', evidence)
    args = ['--stage', 'propagate', '--study', str(tmp_path/'missing_study.yaml'),
            '--batch', 'downloaded', '--matlab', str(matlab), '--toolbox', str(toolbox)]
    return chain, root, toolbox, args


def test_propagate_stage_uses_saved_runtime_and_stops_failed_source_before_matlab(tmp_path, monkeypatch):
    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    monkeypatch.setattr(chain, 'analyze_main', lambda argv: 2)
    monkeypatch.setattr(chain.subprocess, 'run', lambda *a, **k: pytest.fail('MATLAB started on failed source'))
    assert chain.main(args) == 2
    assert not (root/'final/propagation_validation.json').exists()


def test_propagate_stage_exports_local_path_and_runs_existing_numerical_gate(tmp_path, monkeypatch):
    import json
    from scipy.io import savemat, loadmat
    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    source, result = fixture()
    source['observer_role'] = source['observer_role'].astype(object)
    def analyze(argv):
        assert argv[argv.index('--toolbox')+1] == str(toolbox)
        source['toolbox_path'] = str(toolbox)
        (root/'final').mkdir()
        savemat(root/'final/power_r2_kwave_input.mat', source)
        return 0
    monkeypatch.setattr(chain, 'analyze_main', analyze)
    commands = []
    def matlab(command, **kwargs):
        commands.append(command[-1])
        if 'kwave_handoff(' in command[-1]:
            savemat(root/'final/power_r2_kwave_output.mat', result)
    monkeypatch.setattr(chain.subprocess, 'run', matlab)
    assert chain.main(args) == 0
    saved = json.loads((root/'final/propagation_validation.json').read_text())
    assert saved['propagation_validation_passed'] is True
    assert saved['input_sha256'] and saved['output_sha256']
    assert str(loadmat(root/'final/power_r2_kwave_input.mat', squeeze_me=True)['toolbox_path']) == str(toolbox)
    assert sum('kwave_handoff(' in command for command in commands) == 1
    with pytest.raises(FileExistsError):
        chain.main(args)
    assert sum('kwave_handoff(' in command for command in commands) == 1


def test_propagate_stage_rejects_incomplete_raw_run(tmp_path, monkeypatch):
    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    (root/'raw/power_r2/manifest.json').write_text('{"status":"interrupted","source_id":"original_cloud_runtime"}')
    monkeypatch.setattr(chain, 'analyze_main', lambda *a: pytest.fail('analyzed incomplete run'))
    with pytest.raises(ValueError, match='completed'):
        chain.main(args)


@pytest.mark.parametrize('field,value', [('source_clean', False), ('git_commit', None)])
def test_propagate_stage_requires_committed_clean_simulation_runtime(tmp_path, monkeypatch, field, value):
    import json
    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    path = root/'raw/power_r2/manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(chain, 'analyze_main', lambda *a: pytest.fail('analyzed uncommitted runtime'))
    with pytest.raises(ValueError, match='committed'):
        chain.main(args)


def test_diagnostic_propagation_preserves_failed_source_and_existing_analysis(tmp_path, monkeypatch):
    """Execution may finish without turning a failed scientific check into a pass."""
    import json
    from scipy.io import savemat, loadmat
    from Compute.finite_strip.analyze import analyze_data
    from verification.test_strip_analysis import _manufactured_hankel_case
    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    data, config, summary, boundary, _ = _manufactured_hankel_case()
    analysis, arrays = analyze_data(data, config, summary, boundary)
    analysis['source_validation_passed'] = False
    analysis['checks']['kirchhoff_vs_direct'] = False
    analysis['checks']['boundary_evidence_bound'] = True
    monkeypatch.setattr(chain, 'analyze_saved', lambda *a, **k: (analysis, arrays), raising=False)
    monkeypatch.setattr(chain, 'analyze_main', lambda *a: pytest.fail('rewrote existing analysis'))
    final = root/'final'; final.mkdir()
    existing = final/'power_r2_nearfield.json'
    existing.write_text('retained failed analysis', encoding='utf-8')
    calls = []
    def matlab(command, **kwargs):
        calls.append(command[-1])
        if 'kwave_handoff(' not in command[-1]:
            return
        incoming = loadmat(final/'power_r2_kwave_diagnostic_input.mat', squeeze_me=True)
        assert incoming['source_validation_passed'] == 0
        assert incoming['diagnostic_mode'] == 1
        hat = np.asarray(incoming['expected_grid_kirchhoff_p_hat_Pa']).ravel()
        times = np.arange(1, 129)/(64*20000.)
        # Synthetic solver output exercises routing/labels, not k-Wave physics.
        savemat(final/'power_r2_kwave_diagnostic_output.mat', dict(
            observer_x_m=incoming['expected_grid_observer_x_m'],
            observer_y_m=incoming['expected_grid_observer_y_m'],
            observer_p_hat_Pa=hat, time_s=times,
            observer_pressure_Pa=np.real(hat[:, None]*np.exp(2j*np.pi*20000*times)),
            diagnostic_mode=1, source_validation_passed=0, source_validation_bypassed=1))
    monkeypatch.setattr(chain.subprocess, 'run', matlab)
    assert chain.main(args+['--diagnostic', '--matlab-threads', '4']) == 0
    saved = json.loads((final/'propagation_diagnostic.json').read_text())
    assert saved['status'] == 'completed'
    assert saved['diagnostic_mode'] is True
    assert saved['source_validation_passed'] is False
    assert saved['propagation_validation_passed'] is False
    assert saved['checks']['source_validation'] is False
    assert saved['input_sha256'] and saved['output_sha256']
    assert existing.read_text(encoding='utf-8') == 'retained failed analysis'
    assert not (final/'propagation_validation.json').exists()
    assert sum('kwave_handoff(' in c for c in calls) == 1
    assert any('maxNumCompThreads(4)' in c for c in calls)
    with pytest.raises(FileExistsError):
        chain.main(args+['--diagnostic'])
    assert sum('kwave_handoff(' in c for c in calls) == 1


def test_diagnostic_mode_cannot_start_a_new_lbm_run():
    from Compute.finite_strip import chain
    with pytest.raises(SystemExit):
        chain.main(['--batch', 'unused', '--diagnostic', '--stage', 'all'])


def test_diagnostic_output_cannot_change_failed_source_to_passed():
    source,result=fixture()
    source.update(diagnostic_mode=1,source_validation_passed=0)
    result.update(diagnostic_mode=1,source_validation_passed=1,source_validation_bypassed=0)
    with pytest.raises(ValueError,match='metadata'):
        validate_propagation(source,result)
