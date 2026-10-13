"""Real-device parity and invariants for the optional float64 CUDA path."""
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from core.collision_smrt import collide_fg, validate_collision_backend
from core.equilibrium import EquilibriumDomainError, equilibrium_fg
from core.lattice import make_lattice
from core.macroscopic import recover_macro
from core.unit_mapping import create_unit_mapping, local_transport_coefficients


@pytest.fixture
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA device unavailable: {exc}')
    if not count:
        pytest.skip('CUDA device unavailable')
    return cupy


def mapping_for(model='sutherland', bulk=False):
    config = load_config(Path(__file__).resolve().parents[1] /
                         'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')
    config['transport']['model'] = model
    if bulk:
        config['collision'].update(bulk_viscosity_policy='specified', nu_b_lu=.01)
    return create_unit_mapping(config)


def moving_nonequilibrium(mapping):
    lattice = make_lattice()
    rng = np.random.default_rng(6143)
    rho = rng.uniform(.91, 1.13, (3, 7))
    theta = mapping.theta_ref_lu * rng.uniform(.94, 1.06, rho.shape)
    u = rng.uniform(-.07, .07, (*rho.shape, 2))
    f, g = equilibrium_fg(rho, u, theta, mapping.lattice.S, lattice)
    # Unrestricted small perturbations exercise measured stress and heat flux;
    # parity alone is supplemented below by independent raw conserved moments.
    f += rng.normal(0, 2e-5, f.shape) * lattice.w
    g += rng.normal(0, 2e-5, g.shape) * lattice.w
    return lattice, f, g


def test_cupy_backend_is_explicitly_accepted():
    assert validate_collision_backend('cupy') == 'cupy'
    with pytest.raises(ValueError):
        validate_collision_backend('cuda_typo')


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
@pytest.mark.parametrize('bulk', [False, True])
def test_gpu_collision_matches_cpu_and_conserves_each_cell(cp, model, bulk):
    mapping = mapping_for(model, bulk)
    lattice, f, g = moving_nonequilibrium(mapping)
    fg, gg = cp.asarray(f), cp.asarray(g)
    actual_f, actual_g, diag = collide_fg(fg, gg, mapping, lattice=lattice,
                                        backend='cupy', return_diagnostics=True)
    assert isinstance(actual_f, cp.ndarray) and isinstance(actual_g, cp.ndarray)
    assert actual_f.dtype == actual_g.dtype == cp.float64
    for backend in ('numpy', 'numba'):
        expected = collide_fg(f, g, mapping, lattice=lattice, backend=backend)
        for actual, wanted in zip((actual_f, actual_g), expected):
            np.testing.assert_allclose(cp.asnumpy(actual), wanted, rtol=3e-12, atol=2e-14)
    af, ag = cp.asnumpy(actual_f), cp.asnumpy(actual_g)
    np.testing.assert_allclose(af.sum(-1), f.sum(-1), rtol=0, atol=2e-14)
    np.testing.assert_allclose((af-f) @ lattice.c, 0, rtol=0, atol=2e-14)
    raw_energy = .5 * np.sum(lattice.c**2, axis=1)
    np.testing.assert_allclose((af-f) @ raw_energy + (ag-g).sum(-1), 0,
                               rtol=0, atol=2e-14)
    for name in ('mass_residual', 'momentum_residual', 'energy_residual'):
        assert isinstance(getattr(diag, name), cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(getattr(diag, name)), 0, atol=2e-14)
    assert not diag.clipping_used
    np.testing.assert_array_equal(cp.asnumpy(fg), f)
    np.testing.assert_array_equal(cp.asnumpy(gg), g)


def test_gpu_recovery_and_equilibrium_match_cpu_on_strided_fields(cp):
    from core.collision_gpu import equilibrium_gpu, recover_macro_gpu
    mapping = mapping_for()
    lattice, f, g = moving_nonequilibrium(mapping)
    f, g = f[:, ::2], g[:, ::2]
    actual = recover_macro_gpu(cp.asarray(f), cp.asarray(g), D=2, S=3, lattice=lattice)
    expected = recover_macro(f, g, D=2, S=3, lattice=lattice)
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=3e-13, atol=2e-14)
    eq_actual = equilibrium_gpu(actual.rho, actual.u, actual.theta, 3, lattice)
    eq_expected = equilibrium_fg(expected.rho, expected.u, expected.theta, 3, lattice)
    for a, b in zip(eq_actual, eq_expected):
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=3e-13, atol=2e-14)


def test_gpu_scalar_equilibrium_keeps_population_shape(cp):
    from core.collision_gpu import equilibrium_gpu, recover_macro_gpu
    lattice = make_lattice()
    actual = equilibrium_gpu(cp.asarray(1.), cp.asarray([.03, -.02]),
                             cp.asarray(lattice.theta_q), 3, lattice)
    expected = equilibrium_fg(1., [.03, -.02], lattice.theta_q, 3, lattice)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape == (37,)
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=3e-13, atol=2e-14)
    macro = recover_macro_gpu(*actual, D=2, S=3, lattice=lattice)
    assert macro.rho.shape == macro.theta.shape == ()
    assert macro.u.shape == (2,)


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
def test_local_transport_remains_on_device_and_matches_mapping(cp, model):
    mapping = mapping_for(model, bulk=True)
    rho = np.array([[.8], [1.2]])
    theta = mapping.theta_ref_lu * np.array([[.92, 1., 1.08]])
    expected = local_transport_coefficients(mapping, rho, theta)
    actual = local_transport_coefficients(mapping, cp.asarray(rho), cp.asarray(theta))
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        assert isinstance(a, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=3e-14, atol=1e-15)
    with pytest.raises(ValueError, match='positive density'):
        local_transport_coefficients(mapping, cp.array([0.]), cp.array([1.]))
    with pytest.raises(ValueError, match='positive temperature'):
        local_transport_coefficients(mapping, cp.array([1.]), cp.array([float('nan')]))


def test_masked_gpu_equilibrium_increment_audits_actual_writes(cp):
    from core.collision_gpu import apply_equilibrium_increment
    from core.collision_numba import apply_equilibrium_increment as cpu_increment
    mapping = mapping_for()
    lattice, f, g = moving_nonequilibrium(mapping)
    macro = recover_macro(f, g, D=2, S=3, lattice=lattice)
    mask = np.indices(macro.rho.shape).sum(0) % 2 == 0
    new_rho, new_u, new_theta = macro.rho*1.002, macro.u*.97, macro.theta*1.001
    # Invalid inactive targets must neither be evaluated nor contaminate data.
    new_theta[~mask] = np.nan
    args = (f, g, macro.rho, macro.u, macro.theta, new_rho, new_u, new_theta, mask)
    expected = cpu_increment(*args, lattice, 3)
    actual = apply_equilibrium_increment(*(cp.asarray(a) for a in args), lattice, 3)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=3e-12, atol=2e-14)
    af, ag, change = map(cp.asnumpy, actual)
    np.testing.assert_array_equal(af[~mask], f[~mask])
    np.testing.assert_array_equal(ag[~mask], g[~mask])
    raw_energy = .5*np.sum(lattice.c**2, axis=1)
    np.testing.assert_allclose(change, [(af-f).sum(),
        np.sum((af-f) @ raw_energy)+(ag-g).sum()], rtol=0, atol=2e-14)
    gpu_args = tuple(cp.asarray(a) for a in args)
    gpu_args[7][0, 0] = 0.
    with pytest.raises(EquilibriumDomainError):
        apply_equilibrium_increment(*gpu_args, lattice, 3)
    np.testing.assert_array_equal(cp.asnumpy(gpu_args[0]), f)
    np.testing.assert_array_equal(cp.asnumpy(gpu_args[1]), g)


@pytest.mark.parametrize('rho,u,theta', [(-1., [0., 0.], 1.),
    (1., [0., 0.], 0.), (1., [np.nan, 0.], 1.), (1., [5., 0.], 1.)])
def test_gpu_equilibrium_rejects_invalid_states_without_clipping(cp, rho, u, theta):
    from core.collision_gpu import equilibrium_gpu
    with pytest.raises(EquilibriumDomainError):
        equilibrium_gpu(cp.asarray(rho), cp.asarray(u), cp.asarray(theta), 3, make_lattice())


def test_explicit_host_conversion_and_array_namespace(cp):
    from core.array_backend import array_module, to_numpy
    a = np.array([1., 2.])
    assert array_module(a, 1.) is np
    assert array_module(a, cp.asarray(a)) is cp
    np.testing.assert_array_equal(to_numpy(cp.asarray(a)), a)
