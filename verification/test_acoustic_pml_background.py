"""Warm-background acoustic source checks; no open-boundary certification."""
import numpy as np
import pytest

from boundary.acoustic_pml import AcousticPML
from core.macroscopic import recover_macro
from verification.test_finite_strip import solver


def warm_uniform_layer(density_policy):
    gas = solver(film_cells=0)
    mapping = gas.mapping
    rho = mapping.lattice.rho_ref_lu
    if density_policy == 'isobaric':
        rho *= 300./315.
    theta = mapping.theta_ref_lu*315./300.
    gas.initialize_from_macro(rho, np.zeros(2), theta)
    pml = AcousticPML(mapping, gas.lattice, gas.ny, gas.nx,
                      cells=5, target_return=.001, alpha_Hz=1000.)
    return gas, pml


@pytest.mark.parametrize('density_policy', ['isobaric', 'isochoric'])
def test_uniform_315K_background_does_not_relax_to_300K(density_policy):
    gas, pml = warm_uniform_layer(density_policy)
    f, g = gas._require_state()
    fn, gn, memory, change = pml.advance(f, g, pml.memory, .5)
    np.testing.assert_allclose(fn, f, rtol=0, atol=3e-16)
    np.testing.assert_allclose(gn, g, rtol=0, atol=3e-16)
    np.testing.assert_allclose(memory, 0, rtol=0, atol=2e-15)
    np.testing.assert_allclose(change, 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize('density_policy', ['isobaric', 'isochoric'])
def test_warm_uniform_acoustic_memory_uses_local_impedance(density_policy):
    """A constant memory has an analytic exponential acoustic response.

    Uniform fields have zero spatial derivatives, so this checks the Euler
    coefficients against thermodynamics without a second derivative solver.
    Both pressure and velocity memory are excited; the two background paths
    distinguish frozen density and frozen bulk modulus independently.
    """
    gas, pml = warm_uniform_layer(density_policy)
    f, g = gas._require_state()
    mapping = gas.mapping
    old = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S,
                        lattice=gas.lattice)
    # A uniform strip of the absorber isolates its local source, not its
    # interface or endpoint closure. Memory represents a preceding wave.
    pml.sigma_x.fill(.08)
    pml.sigma_y.fill(0.)
    decay = .08002
    pml.decay_x.fill(decay)
    pml.decay_y.fill(0.)
    pml.active.fill(True)
    memory = np.zeros_like(pml.memory)
    memory[0].fill(2e-6)  # pressure-gradient history in x
    memory[2].fill(1e-6)  # velocity-divergence history in x
    duration = .5
    integral = -np.expm1(-decay*duration)/decay
    expected_dp = -old.gamma*old.p*1e-6*integral
    expected_du = -2e-6*integral/old.rho
    expected_drho = expected_dp/(old.gamma*old.theta)

    fn, gn, _, change = pml.advance(f, g, memory, duration)
    new = recover_macro(fn, gn, D=mapping.lattice.D, S=mapping.lattice.S,
                        lattice=gas.lattice)
    # The RK4 truncation at decay*duration=.04001 is below these tolerances;
    # a fixed 300 K coefficient gives percent-level errors in the increments.
    np.testing.assert_allclose(new.u[..., 0]-old.u[..., 0], expected_du,
                               rtol=1e-7, atol=2e-13)
    np.testing.assert_allclose(new.p-old.p, expected_dp, rtol=1e-7, atol=2e-13)
    np.testing.assert_allclose(new.rho-old.rho, expected_drho,
                               rtol=1e-7, atol=2e-13)
    assert min(np.min(fn), np.min(gn)) > 0
    kinetic = .5*np.sum(gas.lattice.c**2, axis=1)
    np.testing.assert_allclose(change,
        [np.sum(fn-f), np.sum((fn-f)*kinetic+gn-g)], rtol=0, atol=2e-13)
