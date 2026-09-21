"""Power-run lifecycle and stationarity measured independently of the solver."""
import hashlib
import json

import numpy as np
import pytest


def synthetic_history(cycles=12, frequency=20000.):
    dt = 1/(frequency*128)
    t = np.arange(1, cycles*128+1)*dt
    tq = t-dt/2
    return dict(t_s=t, heat_time_s=tq,
        film_temperature_K=300+1000*t+.1*np.cos(2*np.pi*frequency*t+3.14),
        wall_flux_W_m2=500+200*t+10*np.cos(2*np.pi*frequency*tq-.4),
        pressure_Pa=(100000+30*t+.3*np.cos(2*np.pi*frequency*t-1))[:,None],
        velocity_m_s=(.001+2*t+.0003*np.cos(2*np.pi*frequency*t-1))[:,None])


def test_stationarity_separates_linear_warming_and_midpoint_phase():
    from postproc.power import stationarity_at_cycle
    from simulation.study import POWER_DEFAULTS

    data=synthetic_history()
    result=stationarity_at_cycle(data, 20000., 12, POWER_DEFAULTS['analysis'])
    assert result['passed']
    temperature=result['channels']['film_temperature']
    assert temperature['amplitude'] == pytest.approx(.1, rel=1e-9)
    assert temperature['drift_per_s'] == pytest.approx(1000., rel=1e-9)
    assert temperature['phase_deg'] == pytest.approx(np.degrees(3.14), abs=1e-7)
    assert result['channels']['wall_flux']['phase_deg'] == pytest.approx(np.degrees(-.4), abs=1e-7)
    assert not stationarity_at_cycle(data,20000.,6,POWER_DEFAULTS['analysis'])['passed']


def test_plot_acceptance_metric_does_not_hide_failed_channel():
    from Compute.open_planar.plot_e1 import stationarity_ratio
    from postproc.power import stationarity_at_cycle
    from simulation.study import POWER_DEFAULTS

    data = synthetic_history()
    good = stationarity_at_cycle(data, 20000., 12, POWER_DEFAULTS['analysis'])
    assert stationarity_ratio(good, POWER_DEFAULTS['analysis']) < 1
    data['velocity_m_s'][:] = 0.
    failed = stationarity_at_cycle(data, 20000., 12, POWER_DEFAULTS['analysis'])
    assert not failed['passed']
    assert np.isnan(stationarity_ratio(failed, POWER_DEFAULTS['analysis']))


@pytest.mark.parametrize('failure', ['amplitude', 'phase', 'zero', 'nonfinite', 'curved_background'])
def test_stationarity_rejects_drift_degeneracy_and_window_bias(failure):
    from postproc.power import stationarity_at_cycle
    from simulation.study import POWER_DEFAULTS

    data=synthetic_history()
    t=data['t_s']; phase=2*np.pi*20000*t
    if failure == 'amplitude':
        data['pressure_Pa'][:,0]=1e5+.3*(1+200*t)*np.cos(phase)
    elif failure == 'phase':
        data['pressure_Pa'][:,0]=1e5+.3*np.cos(phase+1000*t)
    elif failure == 'zero':
        data['velocity_m_s'][:]=0.
    elif failure == 'nonfinite':
        data['pressure_Pa'][-100,0]=np.nan
    else:
        data['film_temperature_K'] += 2e8*t**2
    result=stationarity_at_cycle(data,20000.,12,POWER_DEFAULTS['analysis'])
    assert not result['passed']
    json.dumps(result, allow_nan=False)


def test_power_case_limits_duration_keeps_four_files_and_reanalyses_read_only(tmp_path):
    from simulation import prepare_case, run_case, analyze_case

    config=prepare_case(dict(name='power',mode='power',refine=1,length_m=30e-6,
        frequency_Hz=200000.,cycles=1.,probes_m=[8e-6,18e-6],
        stopping={'stop_when_stable':False}))
    output=tmp_path/'power'
    result=run_case(config,output)
    assert result['status']=='completed'
    assert result['validation_passed'] is False
    assert result['stop_reason']=='cycle_limit'
    assert result['electrical_frequency_Hz']==100000.
    assert {p.name for p in output.iterdir()}=={'effective.yaml','manifest.json','trajectory.npz','summary.json'}
    with np.load(output/'trajectory.npz') as data:
        dt=config['lattice']['dt_s']
        t=data['t_s']; tq=data['heat_time_s']
        np.testing.assert_allclose(t-tq,dt/2,rtol=0,atol=1e-20)
        np.testing.assert_allclose(data['power_W_m2'],1.13/.0009*(1-np.cos(2*np.pi*200000*tq)),atol=1e-10)
        assert data['velocity_m_s'].shape==data['pressure_Pa'].shape
        assert len(data['profile_t_s']) >= 32
        assert data['profile_temperature_K'].shape==data['profile_velocity_m_s'].shape
        assert data['initial_film_temperature_K']==300.
        assert result['max_total_energy_residual_J_m2'] < 1e-10
        assert result['cumulative_total_energy_residual_J_m2'] < 1e-8
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    replay=analyze_case(output)
    assert replay==result
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        run_case(config,output)


@pytest.mark.parametrize('bad', [
    {'amplitude_K':.1}, {'ramp_cycles':2}, {'power':{'area_m2':0}},
    {'film':{'C_A_J_m2K':float('nan')}}, {'film':{'gas_flux_factor':1}},
    {'stopping':{'stop_when_stable':1}}, {'probes_m':[0.]},
])
def test_power_schema_rejects_ambiguous_or_unphysical_inputs(bad):
    from simulation import prepare_case
    with pytest.raises(ValueError):
        prepare_case(dict(name='power',mode='power',**bad))


def test_power_failure_saves_partial_endpoints_and_actual_exchanges(tmp_path,monkeypatch):
    from simulation import prepare_case, run_case
    from core.planar_open_solver import PlanarOpenSolver
    original=PlanarOpenSolver.step_wall_energy

    def fail(self, callback):
        if self.t_lu==3:
            raise RuntimeError('injected root failure')
        return original(self,callback)

    monkeypatch.setattr(PlanarOpenSolver,'step_wall_energy',fail)
    config=prepare_case(dict(name='power',mode='power',refine=1,length_m=30e-6,
        frequency_Hz=200000.,cycles=1.,probes_m=[8e-6,18e-6],stopping={'stop_when_stable':False}))
    with pytest.raises(RuntimeError,match='injected root failure'):
        run_case(config,tmp_path/'power')
    with np.load(tmp_path/'power/trajectory.npz') as data:
        assert len(data['t_s'])==3
        assert data['audit'].shape==(3,6)
        assert not bool(data['run_completed'])
    manifest=json.loads((tmp_path/'power/manifest.json').read_text())
    assert manifest['status']=='failed'


def test_collector_stops_only_at_first_eligible_diagnostic(tmp_path,monkeypatch):
    from simulation import prepare_case, run_case
    checks=[]

    def already_stationary(data,frequency,cycle,analysis):
        # Controlled diagnostic isolates the collector's termination contract;
        # stationarity physics is independently exercised by synthetic_history.
        checks.append(cycle)
        return {'passed':True}

    monkeypatch.setattr('simulation.power.stationarity_at_cycle',already_stationary)
    config=prepare_case(dict(name='power',mode='power',refine=1,length_m=30e-6,
        frequency_Hz=2e6,cycles=24,probes_m=[8e-6,18e-6]))
    result=run_case(config,tmp_path/'power')
    assert checks==[6,7,8]
    assert result['steps']==int(np.ceil(8/2e6/config['lattice']['dt_s']))
    assert result['steps'] < config['numerics']['steps']
    assert result['stop_reason']=='periodic_stable'
