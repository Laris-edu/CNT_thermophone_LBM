"""Cloud launch policy without starting a production LBM calculation."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _boundary_proof():
    return dict(boundary_validation_passed=True, limit=.01, max_return_error_ratio=.005,
                components={label: dict(boundary_validation_passed=True)
                            for label in ('combined', 'top', 'sides')})


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_dry_run_configures_children_without_importing_numba_or_creating_results(tmp_path):
    batch = 'cloud_preview_no_compute'
    # Use the real case expansion; the parent must stay free of numerical imports.
    script = (
        'import sys; from Compute.finite_strip import cloud; '
        "status=cloud.main(['--batch','" + batch + "','--stage','power','--dry-run']); "
        "assert 'numba' not in sys.modules; assert 'numpy' not in sys.modules; "
        'raise SystemExit(status)'
    )
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '"NUMBA_NUM_THREADS": "64"' in result.stdout
    assert '"CNT_LBM_PARALLEL_MIN_CELLS": "131072"' in result.stdout
    assert '"OPENBLAS_NUM_THREADS": "1"' in result.stdout
    assert '"name": "power_r2"' in result.stdout
    assert '"steps":' in result.stdout and '"output":' in result.stdout
    assert not (ROOT/'Compute/finite_strip'/batch).exists()


def test_real_launch_refuses_more_threads_than_available_before_any_child(monkeypatch):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.setattr(cloud.subprocess, 'run', lambda *a, **k: pytest.fail('child started'))
    with pytest.raises(ValueError, match='64.*16'):
        cloud.main(['--batch', 'cloud_capacity_check'])


def test_boundary_preflights_every_case_then_runs_and_compares_with_fixed_environment(monkeypatch, tmp_path):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    calls = []
    def child(command, **kwargs):
        calls.append((command, kwargs))
        if '--compare-pulses' in command:
            _save_json(tmp_path/'fresh/final/boundary_validation.json', _boundary_proof())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cloud.subprocess, 'run', child)
    assert cloud.main(['--study', str(tmp_path/'study.yaml'), '--batch', 'fresh',
                       '--threads', '16']) == 0
    previews = [cmd for cmd, _ in calls if '--dry-run' in cmd]
    runs = [cmd for cmd, _ in calls if '--case' in cmd and '--dry-run' not in cmd]
    expected = ['pulse_candidate_dx10um', 'pulse_reference_dx10um',
                'pulse_sides_dx10um', 'pulse_top_dx10um']
    assert [cmd[cmd.index('--case')+1] for cmd in previews] == expected
    assert [cmd[cmd.index('--case')+1] for cmd in runs] == expected
    assert all('--dry-run' in cmd for cmd, _ in calls[:4])
    assert '--compare-pulses' in calls[-1][0]
    for _, kwargs in calls:
        assert kwargs['env']['NUMBA_NUM_THREADS'] == '16'
        assert kwargs['env']['CNT_LBM_PARALLEL_MIN_CELLS'] == '131072'
        assert kwargs['env']['OPENBLAS_NUM_THREADS'] == kwargs['env']['MKL_NUM_THREADS'] == '1'
        assert kwargs['env']['OMP_NUM_THREADS'] == '16'
        assert kwargs['env']['OMP_DYNAMIC'] == 'FALSE'
        assert kwargs['check'] is True


def test_existing_boundary_proof_prevents_running_new_pulses(monkeypatch, tmp_path):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    proof = tmp_path/'fresh/final/boundary_validation.json'
    proof.parent.mkdir(parents=True)
    proof.write_text('{}')
    calls = []
    monkeypatch.setattr(cloud.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    with pytest.raises(FileExistsError):
        cloud.main(['--study', str(tmp_path/'study.yaml'), '--batch', 'fresh', '--threads', '16'])
    assert all('--dry-run' in cmd for cmd in calls)


def test_conflicting_imported_numba_pool_is_rejected(monkeypatch):
    from Compute.finite_strip import cloud
    monkeypatch.setitem(sys.modules, 'numba', SimpleNamespace(
        config=SimpleNamespace(NUMBA_NUM_THREADS=8), get_num_threads=lambda: 8))
    with pytest.raises(RuntimeError, match='Numba'):
        cloud.main(['--batch', 'pool_conflict', '--threads', '16', '--dry-run'])


@pytest.mark.parametrize('option', ['--threads', '--parallel-min-cells'])
def test_nonpositive_runtime_limits_are_rejected(option):
    from Compute.finite_strip import cloud
    with pytest.raises(SystemExit) as error:
        cloud.main(['--batch', 'invalid_runtime', option, '0', '--dry-run'])
    assert error.value.code == 2


@pytest.mark.parametrize('damage', ['overall', 'component', 'missing_component', 'large_limit',
                                   'zero_limit', 'at_limit', 'negative_error', 'nan_error'])
def test_boundary_zero_exit_does_not_hide_failed_saved_validation(monkeypatch, tmp_path, damage):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    proof = _boundary_proof()
    if damage == 'overall':
        proof['boundary_validation_passed'] = False
    elif damage == 'component':
        proof['components']['sides']['boundary_validation_passed'] = False
    elif damage == 'missing_component':
        del proof['components']['top']
    elif damage == 'large_limit':
        proof['limit'] = .02
    elif damage == 'zero_limit':
        proof['limit'] = 0.
    elif damage == 'at_limit':
        proof['max_return_error_ratio'] = .01
    elif damage == 'negative_error':
        proof['max_return_error_ratio'] = -.001
    else:
        proof['max_return_error_ratio'] = float('nan')
    path = tmp_path/'fresh/final/boundary_validation.json'
    def child(command, **kwargs):
        if '--compare-pulses' in command:
            _save_json(path, proof)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cloud.subprocess, 'run', child)
    assert cloud.main(['--study', str(tmp_path/'study.yaml'), '--batch', 'fresh',
                       '--threads', '16']) != 0
    assert path.read_text(encoding='utf-8') == json.dumps(proof)


@pytest.mark.parametrize('damage', ['manifest_status', 'summary_status', 'state', 'periodic', 'public_false'])
def test_power_zero_exit_does_not_hide_failed_saved_nearfield_gate(monkeypatch, tmp_path, damage):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    manifest = dict(status='completed')
    summary = dict(status='completed', state_validation_passed=True,
                   periodic_response_stable=True, validation_passed=None)
    if damage == 'manifest_status':
        manifest['status'] = 'failed'
    elif damage == 'summary_status':
        summary['status'] = 'interrupted'
    elif damage == 'state':
        summary['state_validation_passed'] = False
    elif damage == 'periodic':
        summary['periodic_response_stable'] = False
    else:
        summary['validation_passed'] = False
    raw = tmp_path/'fresh/raw/power_r2'
    commands = []
    def child(command, **kwargs):
        commands.append(command)
        if '--dry-run' not in command:
            _save_json(raw/'manifest.json', manifest)
            _save_json(raw/'summary.json', summary)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cloud.subprocess, 'run', child)
    assert cloud.main(['--study', str(tmp_path/'study.yaml'), '--batch', 'fresh',
                       '--stage', 'power', '--threads', '16']) != 0
    assert len(commands) == 2
    assert json.loads((raw/'summary.json').read_text()) == summary


def test_power_reports_public_validation_as_unchecked_when_state_and_periodic_gate_pass(monkeypatch, tmp_path, capsys):
    from Compute.finite_strip import cloud
    monkeypatch.setattr(cloud, 'available_cpus', lambda: 16)
    monkeypatch.delitem(sys.modules, 'numba', raising=False)
    raw = tmp_path/'fresh/raw/power_r2'
    summary = dict(status='completed', state_validation_passed=True,
                   periodic_response_stable=True, validation_passed=None)
    def child(command, **kwargs):
        if '--dry-run' not in command:
            _save_json(raw/'manifest.json', dict(status='completed'))
            _save_json(raw/'summary.json', summary)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cloud.subprocess, 'run', child)
    assert cloud.main(['--study', str(tmp_path/'study.yaml'), '--batch', 'fresh',
                       '--stage', 'power', '--threads', '16']) == 0
    output = capsys.readouterr().out
    assert '"validation_passed": null' in output
    assert '"nearfield_state_periodic_gate_passed": true' in output
    assert '"source_validation_passed": null' in output
