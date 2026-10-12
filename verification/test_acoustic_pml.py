"""Tests for the explicitly audited acoustic CFS-PML source, not certification."""
import importlib

import numpy as np
import pytest

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from verification.test_finite_strip import solver


def layer(gas):
    module = importlib.import_module('boundary.acoustic_pml')
    return module.AcousticPML(gas.mapping, gas.lattice, gas.ny, gas.nx,
        cells=5, target_return=.001, alpha_Hz=1000.)


def test_derivatives_include_correct_nonperiodic_edges():
    module = importlib.import_module('boundary.acoustic_pml')
    y, x = np.mgrid[:12, :14]
    field = x*x + 2*y*y + 3*x - y
    np.testing.assert_allclose(module.derivative(field, 1), 2*x+3, atol=1e-13)
    np.testing.assert_allclose(module.derivative(field, 0), 4*y-1, atol=1e-13)


def test_source_preserves_static_entropy_and_quiet_equilibrium():
    gas = solver(film_cells=0)
    pml = layer(gas)
    rho = gas.mapping.lattice.rho_ref_lu*(1+1e-4*np.sin(np.arange(gas.nx))[None,:])
    rho = np.broadcast_to(rho, (gas.ny,gas.nx)).copy()
    pressure = gas.mapping.lattice.rho_ref_lu*gas.mapping.theta_ref_lu
    f,g = equilibrium_fg(rho,np.zeros((*rho.shape,2)),pressure/rho,
                         gas.mapping.lattice.S,gas.lattice)
    fn,gn,memory,change = pml.advance(f,g,pml.memory,.5)
    np.testing.assert_allclose(fn,f,rtol=0,atol=2e-16)
    np.testing.assert_allclose(gn,g,rtol=0,atol=2e-16)
    np.testing.assert_allclose(memory,0,rtol=0,atol=2e-15)
    np.testing.assert_allclose(change,0,rtol=0,atol=1e-12)


def test_source_changes_only_acoustic_macros_and_retains_nonequilibrium():
    gas = solver(film_cells=0)
    pml = layer(gas)
    m=gas.mapping
    y,x=np.mgrid[:gas.ny,:gas.nx]
    gas.initialize_from_macro(m.lattice.rho_ref_lu*(1+1e-5*x),
        np.stack((2e-5*x,-1e-5*y),axis=-1),m.theta_ref_lu*(1+4e-6*x))
    gas.step_temperature(300.)
    f,g=gas._require_state()
    old=recover_macro(f,g,S=m.lattice.S,lattice=gas.lattice)
    fn,gn,memory,change=pml.advance(f,g,pml.memory,.5)
    new=recover_macro(fn,gn,S=m.lattice.S,lattice=gas.lattice)
    eq0=equilibrium_fg(old.rho,old.u,old.theta,m.lattice.S,gas.lattice)
    eq1=equilibrium_fg(new.rho,new.u,new.theta,m.lattice.S,gas.lattice)
    for initial,final,a,b in zip((f,g),(fn,gn),eq0,eq1):
        np.testing.assert_allclose(final-b,initial-a,rtol=0,atol=6e-16)
    np.testing.assert_allclose(new.p-old.p,m.gamma_from_degrees*old.theta*(new.rho-old.rho),atol=6e-16)
    active=pml.active
    np.testing.assert_array_equal(fn[~active],f[~active])
    assert np.max(abs(memory))>0
    kinetic=.5*np.sum(gas.lattice.c**2,axis=1)
    np.testing.assert_allclose(change,[np.sum(fn-f),np.sum((fn-f)*kinetic+gn-g)],atol=1e-13)


def test_source_time_integration_converges_without_population_clipping():
    gas=solver(film_cells=0)
    pml=layer(gas)
    y,x=np.mgrid[:gas.ny,:gas.nx]
    m=gas.mapping
    gas.initialize_from_macro(m.lattice.rho_ref_lu*(1+1e-5*np.sin(x/4)),
        np.stack((1e-5*np.cos(x/4),1e-5*np.sin(y/3)),axis=-1),m.theta_ref_lu)
    answers=[]
    for steps in (1,2,4):
        f,g=gas.f.copy(),gas.g.copy(); memory=pml.memory.copy()
        for _ in range(steps):
            f,g,memory,_=pml.advance(f,g,memory,.25/steps)
        answers.append(np.r_[f.ravel(),g.ravel(),memory.ravel()])
    coarse=np.linalg.norm(answers[0]-answers[2])
    fine=np.linalg.norm(answers[1]-answers[2])
    assert coarse > 1e-14
    assert fine < coarse/8


def test_numpy_and_numba_pml_agree_for_heated_nonequilibrium_half_step():
    """Compiled equilibrium replacement must retain stress/heat-flow content."""
    from boundary.acoustic_pml import AcousticPML

    gas = solver(film_cells=8, backend='numba')
    temperatures = 300.+np.linspace(.01, .08, gas.film_x_m.size)
    for _ in range(12):
        gas.step_temperature(temperatures)
    f, g = gas._require_state()
    macro = gas.get_macro()
    equilibrium = equilibrium_fg(macro.rho, macro.u, macro.theta,
                                gas.mapping.lattice.S, gas.lattice)
    assert max(np.max(abs(f-equilibrium[0])), np.max(abs(g-equilibrium[1]))) > 1e-8
    results = []
    for backend in ('numpy', 'numba'):
        pml = AcousticPML(gas.mapping, gas.lattice, gas.ny, gas.nx,
            cells=5, target_return=.001, alpha_Hz=1000., backend=backend)
        # The second half step exercises accumulated memory, not just startup.
        first_f, first_g, memory, _ = pml.advance(f, g, pml.memory, .5)
        result = pml.advance(first_f, first_g, memory, .5)
        assert np.max(abs(result[2])) > 0
        results.append(result)
    for numpy_value, numba_value in zip(results[0][:3], results[1][:3]):
        np.testing.assert_allclose(numba_value, numpy_value, rtol=3e-13, atol=3e-15)
    np.testing.assert_allclose(results[1][3], results[0][3], rtol=3e-13, atol=3e-13)


def test_solver_pml_exchange_closes_and_failed_callback_rolls_back_memory():
    from core.finite_strip_solver import FiniteStripSolver
    original=solver(film_cells=8)
    gas=FiniteStripSolver(original.config,film_width_m=original.film_dx_m*8,
        pml_cells=5,pml_target_return=.001,pml_alpha_Hz=1000.)
    m=gas.mapping; dx=m.lattice.dx_m
    initial=gas._totals(*gas._require_state())
    exchange=np.zeros(2)
    for _ in range(12):
        a=gas.step_temperature(300.01)
        exchange += [a.outer_mass_lu+a.pml_mass_lu+sum(a.wall_mass_lu),
            (a.outer_energy_J_m+a.pml_energy_J_m+sum(a.wall_energy_J_m2)*dx)/(m.pressure_scale*dx*dx)]
        assert abs(a.energy_residual_J_m)<2e-16
    np.testing.assert_allclose(gas._totals(gas.f,gas.g)-initial,exchange,atol=5e-12)
    f,g,mem=gas.f.copy(),gas.g.copy(),gas.pml.memory.copy()
    def fail(exchange):
        raise RuntimeError('failed root')
    with pytest.raises(RuntimeError,match='failed root'):
        gas.step_wall_energy(fail)
    np.testing.assert_array_equal(gas.f,f)
    np.testing.assert_array_equal(gas.g,g)
    np.testing.assert_array_equal(gas.pml.memory,mem)


def test_complete_cylindrical_pulse_stays_below_one_percent_return_error():
    """Exercise top/side propagation against a newly advanced remote boundary.

    This short coarse-grid regression is not a 20 kHz or thermal-layer test.
    The no-PML control must fail the same 1% gate, so disabling the correction
    cannot silently pass. The reference has no PML and shares no boundary
    histories with either candidate.
    """
    from core.finite_strip_solver import FiniteStripSolver
    from scipy.interpolate import RegularGridInterpolator
    from simulation.finite_strip import prepare_strip_case

    dx, sigma, radius, steps = 10e-6, 45e-6, .00015, 160
    angles = np.linspace(.04, np.pi-.04, 13)
    probe_x = radius*np.cos(angles)
    probe_y = radius*np.sin(angles)
    points = np.column_stack((probe_y, probe_x))
    histories = []
    for width, height, cells in ((.0012, .0003, 12),
                                  (.0012, .0003, 0),
                                  (.0024, .0012, 0)):
        config = prepare_strip_case(dict(kind='pulse', dx_m=dx,
            width_m=width, height_m=height, sponge_width_m=0., sponge_rate=0.))
        gas = FiniteStripSolver(config, film_width_m=0., pml_cells=cells,
                              pml_target_return=.001, pml_alpha_Hz=1000.)
        m = gas.mapping
        p0 = m.lattice.rho_ref_lu*m.theta_ref_lu*m.pressure_scale
        x, y = np.meshgrid(gas.x_m, gas.y_m)
        ratio = 1+np.exp(-.5*(x*x+y*y)/sigma**2)/p0
        gas.initialize_from_macro(m.lattice.rho_ref_lu*ratio**(1/m.gamma_from_degrees),
            np.zeros((*ratio.shape, 2)), m.theta_ref_lu*ratio**(1-1/m.gamma_from_degrees))
        history = []
        for _ in range(steps):
            audit = gas.step_temperature(300.)
            assert abs(audit.energy_residual_J_m) < 1e-10
            pressure = gas.get_macro().p*m.pressure_scale-p0
            history.append(RegularGridInterpolator((gas.y_m, gas.x_m), pressure)(points))
        histories.append(np.asarray(history))
        assert np.isfinite(histories[-1]).all()
        assert np.min(gas.f) > 0 and np.min(gas.g) > 0

    t = np.arange(1, steps+1)*m.lattice.dt_s
    # Six Gaussian widths discount the reference travel distance. Its nearest
    # boundary is 1.2 mm from the source; it cannot return in this time window.
    reference_return = (2*.0012-radius-6*sigma)/m.physical.c0_m_s
    assert t[-1] < reference_return
    assert t[-1] > (2*.0006+radius+3*sigma)/m.physical.c0_m_s
    reference = histories[2]
    scale = np.max(abs(reference), axis=0)
    assert np.min(scale) > .1  # A resolved incident pulse at every probe.
    selected = t >= 1e-6
    pml_error = np.max(abs(histories[0][selected]-reference[selected])/scale)
    old_error = np.max(abs(histories[1][selected]-reference[selected])/scale)
    assert old_error > .01, f'control does not expose the boundary defect: {old_error:.6%}'
    assert pml_error < .01, f'PML returned pressure error is {pml_error:.6%}'
