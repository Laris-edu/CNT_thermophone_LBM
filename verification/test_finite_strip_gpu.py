"""Actual CUDA integration: link boundaries, conservative film and saved data."""
from copy import deepcopy

import numpy as np
import pytest

from core.finite_strip_solver import FiniteStripSolver
from simulation.planar import make_solver
from simulation.finite_strip import prepare_strip_case, run_strip_case


@pytest.fixture
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device unavailable')
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    return cupy


def gas(backend, pml_cells=3, film_cells=12):
    config = deepcopy(make_solver(length=80e-6).config)
    config['numerics'].update(nx=32, ny=18, collision_backend=backend)
    return FiniteStripSolver(config, film_width_m=film_cells*config['lattice']['dx_m'],
                             pml_cells=pml_cells)


def test_gpu_open_stream_preserves_every_link_and_nan_sentinel(cp):
    from boundary.finite_strip import pull_stream_open_xy
    solver = gas('numba')
    f = np.arange(18*32*37, dtype=float).reshape(18, 32, 37)
    cpu = pull_stream_open_xy(f, lattice=solver.lattice)
    gpu = pull_stream_open_xy(cp.asarray(f), lattice=solver.lattice)
    assert isinstance(gpu, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)


@pytest.mark.parametrize('film_cells', [0, 12])
@pytest.mark.parametrize('pml_cells', [0, 3])
def test_gpu_full_steps_match_cpu_with_actual_energy_audit(cp, film_cells, pml_cells):
    cpu, gpu = [gas(backend, pml_cells, film_cells) for backend in ('numba', 'cupy')]
    m = cpu.mapping
    perturb = 2e-5*np.exp(-((cpu.x_m[None, :]/(5*m.lattice.dx_m))**2
                           +((cpu.y_m[:, None]-6*m.lattice.dx_m)/(4*m.lattice.dx_m))**2))
    for solver in (cpu, gpu):
        solver.initialize_from_macro(m.lattice.rho_ref_lu*(1+perturb), np.zeros(2),
                                     m.theta_ref_lu*(1+perturb)**.4)
    assert isinstance(gpu.f, cp.ndarray)
    initial = gpu.get_macro()
    energy0 = float(cp.sum(initial.E_tot))*m.pressure_scale*m.lattice.dx_m**2
    exchange = 0.
    for _ in range(12):
        temp = 300+np.linspace(.01, .08, film_cells)
        left, right = cpu.step_temperature(temp), gpu.step_temperature(temp)
        assert isinstance(right.wall_energy_J_m2, np.ndarray)
        np.testing.assert_allclose(right.wall_energy_J_m2, left.wall_energy_J_m2,
                                   rtol=2e-8, atol=2e-14)
        np.testing.assert_allclose(right.wall_mass_lu, 0, atol=3e-14)
        assert abs(right.energy_residual_J_m) < 2e-15
        exchange += (right.wall_energy_J_m2.sum()*gpu.film_dx_m + right.outer_energy_J_m
                     + right.pml_energy_J_m + right.sponge_energy_J_m)
    for actual, expected in ((gpu.f, cpu.f), (gpu.g, cpu.g)):
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-10, atol=3e-13)
    energy = float(cp.sum(gpu.get_macro().E_tot))*m.pressure_scale*m.lattice.dx_m**2
    assert abs(energy-energy0-exchange) < 3e-15


def test_gpu_failed_wall_trial_preserves_distributions_memory_and_clock(cp):
    solver = gas('cupy')
    solver.step_temperature(300.01)
    before = [cp.asnumpy(v.copy()) for v in (solver.f, solver.g, solver.pml.memory)]
    previous_time, previous_audit = solver.t_lu, solver.last_audit
    def fail(exchange):
        trial = exchange(np.full(12, 300.2))
        assert isinstance(trial, np.ndarray)
        raise RuntimeError('failed film solve')
    with pytest.raises(RuntimeError, match='failed film solve'):
        solver.step_wall_energy(fail)
    for actual, expected in zip((solver.f, solver.g, solver.pml.memory), before):
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)
    assert solver.t_lu == previous_time and solver.last_audit is previous_audit


def test_gpu_conservative_power_and_lateral_film_conduction_match_cpu(cp):
    from coupling.film_ode import FilmOdeParams
    from coupling.finite_strip import solve_strip_film_midpoint, audit_strip_film_step
    solvers = [gas(backend) for backend in ('numba', 'cupy')]
    temperatures = [300+.02*np.cos(np.linspace(-np.pi, np.pi, 12)) for _ in solvers]
    initial_temperatures = [value.copy() for value in temperatures]
    initial_gas = [float(solver.get_macro().E_tot.sum())*solver.mapping.pressure_scale
                   *solver.film_dx_m**2 for solver in solvers]
    external_energy = [0., 0.]
    params = FilmOdeParams(C_A_si=.012888, T_ref_K=300.,
                           linear_leak_conductance_si=0., gas_flux_factor=2.)
    for step in range(12):
        power = 1255.5555555555557*(1-np.cos(.7+.31*step))
        exchanges = []
        for index, solver in enumerate(solvers):
            old = temperatures[index]
            roots = []
            def solve(exchange):
                root = solve_strip_film_midpoint(params=params, T_s_K=old,
                    dx_m=solver.film_dx_m, sheet_conductance_W_K=5.7996e-7,
                    dt_s=solver.mapping.lattice.dt_s, power_W_m2=power,
                    exchange_J_m2=exchange, exchange_is_local=True)
                roots.append(root)
                return root
            audit = solver.step_wall_energy(solve)
            new = 2*roots[-1]-old
            film = audit_strip_film_step(params=params, T_old_K=old, T_new_K=new,
                dx_m=solver.film_dx_m, dt_s=solver.mapping.lattice.dt_s,
                power_W_m2=power, wall_energy_J_m2=audit.wall_energy_J_m2)
            assert abs(film.residual_J_m) < 1e-12
            external_energy[index] += film.input_energy_J_m+2*(audit.outer_energy_J_m
                +audit.pml_energy_J_m+audit.sponge_energy_J_m)
            gas_storage = (float(solver.get_macro().E_tot.sum())*solver.mapping.pressure_scale
                           *solver.film_dx_m**2-initial_gas[index])
            film_storage = params.C_A_si*np.sum(new-initial_temperatures[index])*solver.film_dx_m
            assert abs(film_storage+2*gas_storage-external_energy[index]) < 5e-14
            temperatures[index] = new
            exchanges.append(audit.wall_energy_J_m2)
        np.testing.assert_allclose(temperatures[1], temperatures[0], rtol=0, atol=2e-10)
        np.testing.assert_allclose(exchanges[1], exchanges[0], rtol=2e-8, atol=2e-14)


def test_strip_preparation_preserves_requested_backend():
    config = prepare_strip_case(dict(name='gpu_pulse', kind='pulse', collision_backend='cupy'))
    assert config['numerics']['collision_backend'] == 'cupy'


def test_gpu_saved_pulse_has_host_arrays_and_actual_gpu_provenance(cp, tmp_path):
    import json
    config = prepare_strip_case(dict(name='gpu_saved', kind='pulse', collision_backend='cupy',
        dx_m=1e-5, width_m=.00032, height_m=.00018, duration_s=2e-7,
        pulse_sigma_m=2e-5, probe_radius_m=5e-5, pml_width_m=3e-5,
        sponge_width_m=0, sponge_rate=0))
    output = tmp_path/'gpu_saved'
    summary = run_strip_case(config, output)
    assert summary['status'] == 'completed' and summary['state_validation_passed']
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['collision_backend'] == 'cupy'
    assert manifest['runtime']['device_name']
    assert manifest['runtime']['precision'] == 'float64'
    assert manifest['runtime']['device_id'] == cp.cuda.Device().id
    with np.load(output/'trajectory.npz') as data:
        assert np.isfinite(data['final_pressure_perturbation_Pa']).all()
        assert data['run_completed']
