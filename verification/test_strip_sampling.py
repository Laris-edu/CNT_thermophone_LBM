"""Decimated observations must leave gas stepping and independent audits intact."""
from copy import deepcopy

import numpy as np
import pytest

from core.finite_strip_solver import FiniteStripSolver
from simulation.finite_strip import collect_strip, prepare_strip_case


def test_decimated_pulse_recovers_macro_only_at_saved_endpoints(monkeypatch):
    config = prepare_strip_case(dict(name='sampling', kind='pulse', dx_m=1e-5,
        width_m=.0004, height_m=.00016, probe_radius_m=.00004,
        pulse_sigma_m=.00003, sponge_width_m=0., sponge_rate=0.))
    run = deepcopy(config['strip'])
    run.update(steps=10, sample_every=1)
    full = collect_strip(FiniteStripSolver(config, film_width_m=0.), run)
    gas = FiniteStripSolver(config, film_width_m=0.)
    original = gas.get_macro
    calls = []

    def counted_macro():
        calls.append(gas.t_lu)
        return original()

    monkeypatch.setattr(gas, 'get_macro', counted_macro)
    run['sample_every'] = 3
    decimated = collect_strip(gas, run)
    assert calls == [3, 6, 9, 10], 'Unsaved steps must not rebuild full macro arrays'
    for key in ('audit_t_s', 'audit', 'gas_mass_lu', 'gas_energy_J_m',
                'wall_energy_J_m', 'final_pressure_perturbation_Pa',
                'final_temperature_K', 'final_velocity_m_s'):
        np.testing.assert_array_equal(decimated[key], full[key])
    for key in ('t_s', 'pressure_perturbation_Pa', 'velocity_x_m_s', 'velocity_y_m_s'):
        np.testing.assert_array_equal(decimated[key], full[key][[2, 5, 8, 9]])


def test_early_stop_keeps_actual_final_field_when_stop_is_not_a_sampling_step(monkeypatch):
    import postproc.power

    config = prepare_strip_case(dict(name='early_stop_sampling', kind='power', dx_m=5e-5,
        sponge_width_m=0., sponge_rate=0., field_samples_per_cycle=2))
    run = deepcopy(config['strip'])
    run.update(steps=70, frequency_Hz=1/(7.25*config['lattice']['dt_s']))
    gas = FiniteStripSolver(config, film_width_m=.001)
    original = gas.get_macro
    calls = []

    def counted_macro():
        calls.append(gas.t_lu)
        return original()

    monkeypatch.setattr(gas, 'get_macro', counted_macro)
    # Isolate the already-supported stopping decision from its fit criteria.
    monkeypatch.setattr(postproc.power, 'stationarity_at_cycle', lambda *args: {'passed': True})
    result = collect_strip(gas, run)
    assert result['stop_reason'] == 'periodic_response_stable'
    assert gas.t_lu == 58
    assert calls == [*range(4, 57, 4), 58]
    state = original()
    m = gas.mapping
    pressure0 = m.lattice.rho_ref_lu*m.theta_ref_lu*m.pressure_scale
    np.testing.assert_array_equal(result['final_pressure_perturbation_Pa'], state.p*m.pressure_scale-pressure0)
    np.testing.assert_array_equal(result['final_velocity_m_s'], state.u*m.velocity_scale)
    assert len(result['audit_t_s']) == 58


def test_power_saves_exterior_direct_probes_and_surface_density():
    config = prepare_strip_case(dict(name='handoff_sampling', kind='power', dx_m=1e-5,
        sponge_width_m=0., sponge_rate=0., pml_width_m=.0002))
    run = dict(config['strip'], steps=1)
    gas = FiniteStripSolver(config, film_width_m=.001, pml_cells=run['pml_cells'])
    data = collect_strip(gas, run)
    assert len(data['probe_x_m']) == 16
    np.testing.assert_array_equal(data['probe_x_m'][13:], [0., -.00078, .00078])
    np.testing.assert_array_equal(data['probe_y_m'][13:], [.00028, .0001, .0001])
    x, y = data['probe_x_m'][13:], data['probe_y_m'][13:]
    assert np.all((abs(x) > .00075) | (y > .00025))
    assert np.all(abs(x) < .0008) and np.all(y < .0003)
    assert data['surface_channels'].tolist()[-1] == 'rho_kg_m3'
    assert data['surface_fields'].shape[-1] == 5
    assert np.all(data['surface_fields'][..., -1] > 0)


def test_power_rejects_handoff_surface_inside_pml_before_stepping():
    with pytest.raises(ValueError, match='outside'):
        prepare_strip_case(dict(name='pml_margin', kind='power', dx_m=1e-5,
            sponge_width_m=0., sponge_rate=0., pml_width_m=.0003))


def test_early_stop_waits_for_the_entire_handoff_line(monkeypatch):
    import postproc.power
    config=prepare_strip_case(dict(name='handoff_stop',kind='power',dx_m=5e-5,
        sponge_width_m=0.,sponge_rate=0.,field_samples_per_cycle=2))
    run=dict(config['strip'],steps=70,frequency_Hz=1/(7.25*config['lattice']['dt_s']))
    # Film bins and the small set of probes appear stable, but the complete
    # saved contour has not yet settled. It must not be exported early.
    def check(data,*args):
        return {'passed':data['pressure_Pa'].shape[1]<=16,'channels':{}}
    monkeypatch.setattr(postproc.power,'stationarity_at_cycle',check)
    gas=FiniteStripSolver(config,film_width_m=.001)
    result=collect_strip(gas,run)
    assert gas.t_lu==70
    assert result['periodic_response_stable']==False
    assert result['stop_reason']=='maximum_steps'
