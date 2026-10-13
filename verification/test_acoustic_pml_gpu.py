"""Optional CUDA regressions for the shared, float64 acoustic CFS-PML."""
import numpy as np
import pytest

from boundary.acoustic_pml import AcousticPML, derivative
from core.equilibrium import EquilibriumDomainError, equilibrium_fg
from core.macroscopic import recover_macro
from verification.test_finite_strip import solver


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    if not count:
        pytest.skip('CUDA device unavailable')
    return cupy


def layer(gas, backend):
    return AcousticPML(gas.mapping, gas.lattice, gas.ny, gas.nx,
                       cells=5, alpha_Hz=1000., backend=backend)


def population_fixture(gas):
    """Resolved thermal/acoustic variation plus non-equilibrium populations."""
    m = gas.mapping
    y, x = np.mgrid[:gas.ny, :gas.nx]
    rho = m.lattice.rho_ref_lu*(1+.001*np.sin(x/5)+.002*np.cos(y/4))
    u = np.stack((.0003*np.sin(x/4), .0002*np.cos(y/5)), axis=-1)
    theta = m.theta_ref_lu*(1+.003*np.sin((x+y)/6))
    f, g = equilibrium_fg(rho, u, theta, m.lattice.S, gas.lattice)
    rng = np.random.default_rng(29477)
    f += rng.normal(0., 1e-6, f.shape)*gas.lattice.w
    g += rng.normal(0., 1e-6, g.shape)*gas.lattice.w
    return f, g


def test_gpu_derivative_resolves_both_nonperiodic_edge_stencils(cp):
    y, x = np.mgrid[:12, :14]
    field = cp.asarray(x*x+2*y*y+3*x-y, dtype=cp.float64)
    for axis, expected in ((1, 2*x+3), (0, 4*y-1)):
        actual = derivative(field, axis)
        assert isinstance(actual, cp.ndarray) and actual.dtype == cp.float64
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_gpu_rhs_matches_cpu_with_nonuniform_fields_and_corners(cp, backend):
    gas = solver(film_cells=0)
    reference, gpu = layer(gas, backend), layer(gas, 'cupy')
    rng = np.random.default_rng(6483)
    v = rng.normal(0., .02, (7, gas.ny, gas.nx))
    rho = rng.uniform(.95, 1.05, (gas.ny, gas.nx))
    c2 = rng.uniform(.95, 1.05, rho.shape)*reference.c2
    inputs = tuple(cp.asarray(value) for value in (v, rho, c2))
    actual = gpu._rhs(*inputs)
    assert isinstance(actual, cp.ndarray) and actual.dtype == cp.float64
    np.testing.assert_allclose(cp.asnumpy(actual), reference._rhs(v, rho, c2),
                               rtol=3e-13, atol=3e-15)
    for actual_input, expected_input in zip(inputs, (v, rho, c2)):
        np.testing.assert_array_equal(cp.asnumpy(actual_input), expected_input)


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_gpu_half_steps_preserve_nonequilibrium_and_audit_actual_exchange(cp, backend):
    gas = solver(film_cells=0)
    f, g = population_fixture(gas)
    reference, gpu = layer(gas, backend), layer(gas, 'cupy')
    for name in ('sigma_x', 'sigma_y', 'decay_x', 'decay_y', 'active', 'memory'):
        assert isinstance(getattr(gpu, name), cp.ndarray)
    f_gpu, g_gpu = cp.asarray(f), cp.asarray(g)
    memory_gpu = gpu.memory.copy()
    cpu_state = (f, g, reference.memory.copy())
    gpu_state = (f_gpu, g_gpu, memory_gpu)
    kinetic = .5*np.sum(gas.lattice.c**2, axis=1)
    for _ in range(2):
        saved = tuple(cp.asnumpy(value) for value in gpu_state)
        expected = reference.advance(*cpu_state, .5)
        result = gpu.advance(*gpu_state, .5)
        for value in result[:3]:
            assert isinstance(value, cp.ndarray) and value.dtype == cp.float64
        actual = tuple(cp.asnumpy(value) for value in result)
        for value, expected_value in zip(actual[:3], expected[:3]):
            np.testing.assert_allclose(value, expected_value, rtol=3e-13, atol=3e-15)
        np.testing.assert_allclose(actual[3], expected[3], rtol=3e-13, atol=3e-13)
        # Independent audit from stored populations, not the source RHS.
        df, dg = actual[0]-saved[0], actual[1]-saved[1]
        measured = np.array([np.sum(df), np.sum(df*kinetic+dg)])
        np.testing.assert_allclose(actual[3], measured, rtol=3e-13, atol=3e-13)
        active = cp.asnumpy(gpu.active)
        for new, old in zip(actual[:2], saved[:2]):
            np.testing.assert_array_equal(new[~active], old[~active])
        old_macro = recover_macro(*saved[:2], S=gas.mapping.lattice.S, lattice=gas.lattice)
        new_macro = recover_macro(*actual[:2], S=gas.mapping.lattice.S, lattice=gas.lattice)
        old_eq = equilibrium_fg(old_macro.rho, old_macro.u, old_macro.theta,
                               gas.mapping.lattice.S, gas.lattice)
        new_eq = equilibrium_fg(new_macro.rho, new_macro.u, new_macro.theta,
                               gas.mapping.lattice.S, gas.lattice)
        assert max(np.max(abs(saved[j]-old_eq[j])) for j in (0, 1)) > 1e-8
        for old, new, eq0, eq1 in zip(saved[:2], actual[:2], old_eq, new_eq):
            np.testing.assert_allclose(new-eq1, old-eq0, rtol=0, atol=2e-15)
        for initial, unchanged in zip(saved, gpu_state):
            np.testing.assert_array_equal(cp.asnumpy(unchanged), initial)
        cpu_state, gpu_state = expected[:3], result[:3]
    assert np.max(abs(cp.asnumpy(gpu_state[2]))) > 0
    np.testing.assert_array_equal(cp.asnumpy(gpu.memory), 0)


@pytest.mark.parametrize('failure', ['duration', 'nonfinite_memory', 'invalid_equilibrium'])
def test_gpu_failed_half_step_does_not_commit_populations_or_memory(cp, failure):
    gas = solver(film_cells=0)
    gpu = layer(gas, 'cupy')
    inputs = [cp.asarray(value) for value in population_fixture(gas)]
    inputs.append(gpu.memory.copy())
    duration = .5
    error = (EquilibriumDomainError, FloatingPointError)
    if failure == 'duration':
        duration, error = 0., ValueError
    elif failure == 'nonfinite_memory':
        inputs[2][0, -1, 0] = cp.nan
    else:
        # A large momentum source invalidates the H4/H2 expansion.
        inputs[2][0, -1, 0] = 1e4
    saved = tuple(cp.asnumpy(value) for value in inputs)
    with pytest.raises(error):
        gpu.advance(*inputs, duration)
    for before, after in zip(saved, inputs):
        np.testing.assert_array_equal(cp.asnumpy(after), before)
    np.testing.assert_array_equal(cp.asnumpy(gpu.memory), 0)
