"""An exact moment-state path must retain the original physical wall exchanges."""
from dataclasses import astuple

import numpy as np
import pytest

from core.planar_open_solver import PlanarOpenSolver
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from coupling.planar_open import step_open_film
from simulation.study import ROOT, prepare_case


def configuration():
    return prepare_case(dict(name='moment_check', mode='thermal_step', refine=4,
        length_m=128*0.6529375e-6, probes_m=[15e-6, 40e-6, 65e-6],
        amplitude_K=.1, thermal_step=dict(duration_s=1e-7, interface_m=32*0.6529375e-6),
        sampling=dict(probe_interpolation='linear', sync_fine_steps=2)),
        ROOT/'Compute/open_planar/e1_long_air.yaml')


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_moment_state_matches_wall_and_film_history_with_drift(backend):
    from core.moment_planar_solver import MomentPlanarOpenSolver
    from core.kinetic_moments import extract_moments
    config = configuration()
    config['numerics']['collision_backend'] = backend
    reference, compact = PlanarOpenSolver(config), MomentPlanarOpenSolver(config)
    m = reference.mapping
    z = reference.z_m
    perturbation = 1e-5*np.exp(-.5*((z-40e-6)/8e-6)**2)
    rho = m.lattice.rho_ref_lu*(1+perturbation[:, None])
    theta = m.theta_ref_lu*(1+.002+perturbation[:, None]*(m.physical.gamma-1))
    velocity = np.zeros((reference.ny, 1, 2))
    velocity[:, 0, 1] = m.c0_lu*perturbation
    for solver in (reference, compact):
        solver.initialize_from_macro(rho, velocity, theta)
    params = FilmOdeParams(C_A_si=.0007, T_ref_K=300.,
                          linear_leak_conductance_si=57.8, gas_flux_factor=2.)
    drive = SinusoidalDrive(1255.5555555555557, -1255.5555555555557, 20000.)
    temperatures = [300., 300.]
    for _ in range(24):
        records = []
        for index, solver in enumerate((reference, compact)):
            record = step_open_film(solver=solver, params=params,
                                   T_s_K=temperatures[index], drive=drive)
            temperatures[index] = record.T_s_K
            records.append(record)
        np.testing.assert_allclose(compact.moments,
            extract_moments(reference.f, reference.g, reference.lattice), atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(compact.get_heat_flux_lu(), reference.get_heat_flux_lu(),
                                   atol=3e-13, rtol=3e-9)
        np.testing.assert_allclose(astuple(records[0].audit), astuple(records[1].audit),
                                   atol=3e-12, rtol=3e-9)
        assert temperatures[0] == pytest.approx(temperatures[1], abs=3e-9)
    assert compact.f is None and compact.g is None
    assert compact.moments.nbytes == reference.ny*9*8
    assert min(compact.population_extrema()) > 0


def test_failed_wall_solve_preserves_moments_clock_and_audits():
    from core.moment_planar_solver import MomentPlanarOpenSolver
    solver = MomentPlanarOpenSolver(configuration())
    solver.step_temperature(300.1)
    before = solver.checkpoint()
    def fail(exchange):
        exchange(300.2)
        raise RuntimeError('intentional callback failure')
    with pytest.raises(RuntimeError, match='intentional'):
        solver.step_wall_energy(fail)
    after = solver.checkpoint()
    np.testing.assert_array_equal(after['moments'], before['moments'])
    assert after['t_lu'] == before['t_lu']
    assert after['last_audit'] == before['last_audit']


def test_moment_workflow_four_files_and_read_only_reanalysis(tmp_path):
    from simulation import run_case, analyze_case
    import hashlib
    config = configuration()
    # The public selector belongs to the solver, not the physical run parameters.
    parameters = dict(name='compact', mode='thermal_step', refine=4,
        length_m=128*.6529375e-6, probes_m=[15e-6, 40e-6, 65e-6],
        amplitude_K=.1, state_representation='moments9',
        thermal_step=dict(duration_s=20e-9, interface_m=32*.6529375e-6),
        sampling=dict(probe_interpolation='linear', sync_fine_steps=2))
    config = prepare_case(parameters, ROOT/'Compute/open_planar/e1_long_air.yaml')
    output = tmp_path/'compact'
    result = run_case(config, output)
    assert result['state_representation'] == 'moments9'
    assert result['status'] == 'completed'
    assert result['state_storage_bytes'] == config['numerics']['ny']*9*8
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert len(hashes) == 4
    analyzed = analyze_case(output)
    assert analyzed['validation_passed'] == result['validation_passed']
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    with np.load(output/'trajectory.npz') as data:
        assert data['moment_state_final'].shape == (config['numerics']['ny'], 1, 9)
        assert data['representation_equivalence_only']


def test_independent_grid_is_not_silently_accepted_by_moment_representation():
    params = dict(name='invalid', mode='thermal_step', refine=4,
        length_m=128*.6529375e-6, probes_m=[15e-6, 40e-6, 65e-6],
        state_representation='moments9',
        grid=dict(type='two_level', ratio=2, interface_m=32*.6529375e-6),
        thermal_step=dict(duration_s=20e-9, interface_m=32*.6529375e-6),
        sampling=dict(sync_fine_steps=2))
    with pytest.raises(ValueError, match='moments9'):
        prepare_case(params, ROOT/'Compute/open_planar/e1_long_air.yaml')


@pytest.mark.parametrize('representation', ['populations', 'moments9'])
def test_second_fine_step_failure_restores_last_sync_state(tmp_path, representation):
    from core.moment_planar_solver import MomentPlanarOpenSolver
    from simulation.multigrid import collect_synchronized
    config = configuration()
    solver_type = MomentPlanarOpenSolver if representation == 'moments9' else PlanarOpenSolver
    solver = solver_type(config)
    solver.get_macro()
    before = solver.checkpoint()
    step = solver.step_temperature
    calls = 0
    def fail_second(temperature):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError('intentional second-step failure')
        return step(temperature)
    solver.step_temperature = fail_second
    with pytest.raises(FloatingPointError, match='second-step'):
        collect_synchronized(solver, tmp_path, config['run'])
    assert solver.t_lu == 0
    after = solver.checkpoint()
    for key in ('moments',) if representation == 'moments9' else ('f', 'g'):
        np.testing.assert_array_equal(after[key], before[key])
    assert after['last_audit'] == before['last_audit']
    with np.load(tmp_path/'trajectory.npz') as data:
        assert len(data['t_s']) == 0
        assert int(data['cell_updates']) == 0
        assert not bool(data['run_completed'])


@pytest.mark.parametrize('representation', ['populations', 'moments9'])
@pytest.mark.parametrize('failed_readout', [2, 3])
def test_observation_failure_restores_sync(tmp_path, representation, failed_readout):
    """A failed probe readout must not export a state ahead of saved history."""
    from core.moment_planar_solver import MomentPlanarOpenSolver
    from simulation.multigrid import collect_synchronized
    solver_type = MomentPlanarOpenSolver if representation == 'moments9' else PlanarOpenSolver
    config = configuration()
    solver = solver_type(config)
    solver.get_macro()
    expected = solver.checkpoint()
    expected_extrema = None
    readout = solver.get_heat_flux_lu
    calls = 0

    def fail_probe_readout():
        nonlocal calls, expected, expected_extrema
        calls += 1
        # Call 1 is the initial profile; calls 2/3 are synchronized probe reads.
        # This short run has no scheduled intermediate profile snapshots.
        if calls == failed_readout:
            raise RuntimeError('intentional synchronized observation failure')
        value = readout()
        if calls > 1:
            expected = solver.checkpoint()
            expected_extrema = solver.population_extrema()
        return value

    solver.get_heat_flux_lu = fail_probe_readout
    with pytest.raises(RuntimeError, match='synchronized observation failure'):
        collect_synchronized(solver, tmp_path, config['run'])
    after = solver.checkpoint()
    assert after['t_lu'] == expected['t_lu'] == 2*(failed_readout-2)
    for key in ('moments',) if representation == 'moments9' else ('f', 'g'):
        np.testing.assert_array_equal(after[key], expected[key])
    assert after['last_audit'] == expected['last_audit']
    if expected['last_interface_audit'] is None:
        assert after['last_interface_audit'] is None
    else:
        np.testing.assert_array_equal(after['last_interface_audit'], expected['last_interface_audit'])
    with np.load(tmp_path/'trajectory.npz') as data:
        committed = failed_readout-2
        assert len(data['t_s']) == committed
        assert len(data['heat_time_s']) == 2*committed
        assert len(data['conductive_heat_flux_W_m2']) == committed
        assert len(data['audit']) == len(data['block_conserved']) == committed
        assert int(data['cell_updates']) == 2*committed*solver.ny
        np.testing.assert_array_equal(data['level_steps'], [2*committed, 0])
        assert not bool(data['run_completed'])
        if representation == 'moments9':
            np.testing.assert_array_equal(data['moment_state_final'], expected['moments'])
        if expected_extrema is None:
            assert np.isposinf(data['min_f']) and np.isposinf(data['min_g'])
        else:
            np.testing.assert_array_equal([data['min_f'], data['min_g']], expected_extrema)


@pytest.mark.parametrize('representation', ['populations', 'moments9'])
def test_power_second_fine_callback_failure_restores_sync(tmp_path, representation):
    """Discard the partial second gas/film pair, retaining the first real pair."""
    from core.moment_planar_solver import MomentPlanarOpenSolver
    from simulation.multigrid import collect_synchronized
    config = prepare_case(dict(name='power_failure', mode='power', refine=4,
        length_m=128*.6529375e-6, probes_m=[15e-6, 40e-6, 65e-6],
        state_representation=representation, cycles=.001,
        film=dict(initial_temperature_K=300.01),
        stopping=dict(stop_when_stable=False),
        sampling=dict(probe_interpolation='linear', sync_fine_steps=2)),
        ROOT/'Compute/open_planar/e1_long_air.yaml')
    solver_type = MomentPlanarOpenSolver if representation == 'moments9' else PlanarOpenSolver
    solver, reference = solver_type(config), solver_type(config)
    solver.get_macro(); reference.get_macro()
    film = config['run']['film']; power = config['run']['power']
    params = FilmOdeParams(C_A_si=film['C_A_J_m2K'], T_ref_K=film['T_ref_K'],
        linear_leak_conductance_si=film['G_W_m2K'], gas_flux_factor=film['gas_flux_factor'])
    mean = power['mean_W']/power['area_m2']
    drive = SinusoidalDrive(mean, -mean, config['run']['frequency_Hz'])
    temperature = film['initial_temperature_K']; records = []
    for _ in range(2):
        record = step_open_film(solver=reference, params=params, T_s_K=temperature, drive=drive)
        temperature = record.T_s_K; records.append(record)
    expected = reference.checkpoint()
    step_wall = solver.step_wall_energy
    calls = 0

    def fail_fourth(solve):
        nonlocal calls
        calls += 1
        if calls == 4:
            def fail(exchange):
                exchange(300.01)
                raise RuntimeError('intentional second fine film callback failure')
            return step_wall(fail)
        return step_wall(solve)

    solver.step_wall_energy = fail_fourth
    with pytest.raises(RuntimeError, match='second fine film callback failure'):
        collect_synchronized(solver, tmp_path, config['run'])
    actual = solver.checkpoint()
    assert calls == 4 and actual['t_lu'] == expected['t_lu'] == 2
    for key in ('moments',) if representation == 'moments9' else ('f', 'g'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['last_audit'] == expected['last_audit']
    assert actual['last_interface_audit'] is expected['last_interface_audit'] is None
    dt = solver.mapping.lattice.dt_s
    with np.load(tmp_path/'trajectory.npz') as data:
        np.testing.assert_array_equal(data['t_s'], [2*dt])
        np.testing.assert_array_equal(data['film_t_s'], [dt, 2*dt])
        np.testing.assert_array_equal(data['heat_time_s'], [.5*dt, 1.5*dt])
        for key, values in (
            ('film_temperature_K', [r.T_s_K for r in records]),
            ('film_energy_J_m2', [params.C_A_si*(r.T_s_K-params.T_ref_K) for r in records]),
            ('wall_energy_J_m2', [r.audit.wall_energy_J_m2 for r in records]),
            ('input_energy_J_m2', [r.input_energy_J_m2 for r in records]),
            ('leak_energy_J_m2', [r.leak_energy_J_m2 for r in records])):
            np.testing.assert_array_equal(data[key], values)
        expected_audit = np.sum([astuple(r.audit) for r in records], axis=0)
        np.testing.assert_array_equal(data['audit'], expected_audit[None, :])
        macro = reference.get_macro(); m = reference.mapping
        energy = np.sum(macro.E_tot[:, 0]*m.lattice.dx_m)*m.pressure_scale
        np.testing.assert_array_equal(data['gas_energy_J_m2'], [energy])
        assert len(data['pressure_Pa']) == len(data['mass_kg_m2']) == 1
        assert int(data['cell_updates']) == 2*solver.ny
        np.testing.assert_array_equal(data['level_steps'], [2, 0])
        assert not bool(data['run_completed'])
        if representation == 'moments9':
            np.testing.assert_array_equal(data['moment_state_final'], expected['moments'])


def test_zero_perturbation_1000_steps_has_no_added_sound():
    """Use the existing equilibrium test's absolute 5e-14 LU bounds in SI."""
    from core.moment_planar_solver import MomentPlanarOpenSolver
    config = configuration()
    config['numerics']['collision_backend'] = 'numba'
    reference, compact = PlanarOpenSolver(config), MomentPlanarOpenSolver(config)
    m = reference.mapping
    initial_pressure = reference.get_macro().p.copy()
    maxima = np.zeros((3, 2))  # population, moments, and their difference: p/u
    for _ in range(1000):
        for solver in (reference, compact):
            audit = solver.step_temperature(m.physical.T0_K)
            assert abs(audit.wall_mass_lu) < 1e-14
            assert abs(audit.wall_energy_J_m2) < 1e-13
        states = (reference.get_macro(), compact.get_macro())
        for index, state in enumerate(states):
            maxima[index, 0] = max(maxima[index, 0], np.max(abs(state.p-initial_pressure))*m.pressure_scale)
            maxima[index, 1] = max(maxima[index, 1], np.max(abs(state.u))*m.velocity_scale)
        maxima[2, 0] = max(maxima[2, 0], np.max(abs(states[0].p-states[1].p))*m.pressure_scale)
        maxima[2, 1] = max(maxima[2, 1], np.max(abs(states[0].u-states[1].u))*m.velocity_scale)
    assert np.all(maxima[:, 0] <= 5e-14*m.pressure_scale), maxima
    assert np.all(maxima[:, 1] <= 5e-14*m.velocity_scale), maxima
