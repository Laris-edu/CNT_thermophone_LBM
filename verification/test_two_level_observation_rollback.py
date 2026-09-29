"""Two-level physical state and exported observations share a commit boundary."""
from copy import deepcopy
from dataclasses import astuple

import numpy as np
import pytest

from core.two_level_planar_solver import TwoLevelPlanarOpenSolver
from simulation.multigrid import collect_synchronized
from simulation.study import ROOT, prepare_case


def config():
    h = .6529375e-6
    return prepare_case(dict(name='rollback', mode='thermal_step', refine=4,
        length_m=128*h, probes_m=[15e-6, 40e-6, 65e-6], amplitude_K=.1,
        grid=dict(type='two_level', ratio=2, interface_m=32*h),
        thermal_step=dict(duration_s=20e-9, interface_m=32*h),
        sampling=dict(probe_interpolation='linear', sync_fine_steps=2)),
        ROOT/'Compute/open_planar/e1_long_air.yaml')


def snapshot(solver):
    """Independent test capture: do not make assertions through checkpoint()."""
    return dict(arrays=[a.copy() for a in (solver.fine_solver.f, solver.fine_solver.g,
                                         solver.coarse_solver.f, solver.coarse_solver.g)],
        clocks=(solver.t_lu, solver.fine_solver.t_lu, solver.coarse_solver.t_lu),
        audit=deepcopy(solver.last_audit), substeps=deepcopy(solver.last_substep_audits),
        interface=deepcopy(solver.last_interface_audit),
        momentum=deepcopy(solver.last_momentum_residual_kg_m_s),
        level_steps=solver.level_steps.copy(), cell_updates=solver.cell_updates)


@pytest.mark.parametrize('failed_readout', [2, 3])
def test_two_level_observation_failure_restores_sync(tmp_path, failed_readout):
    solver = TwoLevelPlanarOpenSolver(config())
    solver.initialize_equilibrium()
    expected = snapshot(solver)
    expected_extrema = None
    readout = solver.get_heat_flux_lu
    calls = 0

    def fail_probe_readout():
        nonlocal calls, expected, expected_extrema
        calls += 1
        if calls == failed_readout:
            raise RuntimeError('intentional two-level observation failure')
        value = readout()
        if calls > 1:
            expected = snapshot(solver)
            expected_extrema = [float(np.min(solver.f)), float(np.min(solver.g))]
        return value

    solver.get_heat_flux_lu = fail_probe_readout
    with pytest.raises(RuntimeError, match='two-level observation failure'):
        collect_synchronized(solver, tmp_path, solver.config['run'])
    actual = snapshot(solver)
    for a, b in zip(actual.pop('arrays'), expected.pop('arrays')):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual.pop('momentum'), expected.pop('momentum'))
    assert actual == expected
    committed = failed_readout-2
    assert actual['clocks'] == (2*committed, 2*committed, committed)
    with np.load(tmp_path/'trajectory.npz') as data:
        assert len(data['t_s']) == len(data['conductive_heat_flux_W_m2']) == committed
        assert len(data['audit']) == len(data['interface_exchange']) == len(data['block_conserved']) == committed
        assert len(data['heat_time_s']) == len(data['wall_energy_J_m2']) == 2*committed
        assert int(data['cell_updates']) == committed*(2*solver.nf+solver.nc)
        np.testing.assert_array_equal(data['level_steps'], [2*committed, committed])
        assert not bool(data['run_completed'])
        if expected_extrema is None:
            assert np.isposinf(data['min_f']) and np.isposinf(data['min_g'])
        else:
            np.testing.assert_array_equal([data['min_f'], data['min_g']], expected_extrema)
        if committed:
            np.testing.assert_array_equal(data['audit'][-1], astuple(expected['audit']))
            np.testing.assert_array_equal(data['interface_exchange'][-1], expected['interface'].as_array())
            np.testing.assert_array_equal(data['wall_energy_J_m2'],
                [record.wall_energy_J_m2 for record in expected['substeps']])


def test_checkpoint_owns_arrays_and_mutable_audit_state():
    solver = TwoLevelPlanarOpenSolver(config())
    solver.step_temperature(300.1)
    saved = solver.checkpoint()
    originals = deepcopy(saved)
    solver.fine_solver.f[0, 0, 0] += 1
    solver.last_momentum_residual_kg_m_s[:] = 1
    solver.level_steps['fine'] = 999
    np.testing.assert_array_equal(saved['fine_f'], originals['fine_f'])
    np.testing.assert_array_equal(saved['last_momentum_residual_kg_m_s'], originals['last_momentum_residual_kg_m_s'])
    assert saved['level_steps'] == originals['level_steps']
    solver.restore(saved)
    for name, actual in zip(('fine_f', 'fine_g', 'coarse_f', 'coarse_g'),
            (solver.fine_solver.f, solver.fine_solver.g, solver.coarse_solver.f, solver.coarse_solver.g)):
        np.testing.assert_array_equal(actual, originals[name])
        assert not np.shares_memory(actual, saved[name])
    assert not np.shares_memory(solver.last_momentum_residual_kg_m_s, saved['last_momentum_residual_kg_m_s'])
    assert solver.level_steps == originals['level_steps']
    assert solver.level_steps is not saved['level_steps']
