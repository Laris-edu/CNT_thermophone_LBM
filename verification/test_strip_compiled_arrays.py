"""Compiled finite-strip memory passes retain open links and actual storage."""
import importlib
import importlib.util

import numpy as np
import pytest

from core.lattice import make_lattice
from core.finite_strip_solver import FiniteStripSolver
from simulation.finite_strip import prepare_strip_case


@pytest.mark.parametrize('parallel', [False, True])
def test_compiled_pair_stream_matches_open_numpy_links(parallel, monkeypatch):
    import numba
    import core.collision_numba as dispatch
    from boundary.finite_strip import pull_stream_open_xy

    lat = make_lattice()
    f = np.arange(11*26*37, dtype=float).reshape(11, 26, 37)[:, ::2]
    g = -2*f+7
    expected = [pull_stream_open_xy(v, lattice=lat) for v in (f, g)]
    assert importlib.util.find_spec('core.finite_strip_numba') is not None, 'compiled pair streaming is missing'
    compiled = importlib.import_module('core.finite_strip_numba')
    old_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        monkeypatch.setattr(dispatch, 'PARALLEL_MIN_CELLS', 1 if parallel else 10**9)
        actual = compiled.stream_open_xy_pair(f, g, lat.c)
    finally:
        numba.set_num_threads(old_threads)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(f, np.arange(11*26*37, dtype=float).reshape(11, 26, 37)[:, ::2])


@pytest.mark.parametrize('parallel', [False, True])
def test_fused_statistics_read_independent_population_storage(parallel, monkeypatch):
    import numba
    import core.collision_numba as dispatch

    lat = make_lattice()
    rng = np.random.default_rng(891)
    f, g = rng.uniform(.01, 1., (2, 17, 21, 37))
    kinetic = .5*np.sum(lat.c**2, axis=1)
    assert importlib.util.find_spec('core.finite_strip_numba') is not None, 'fused population statistics are missing'
    compiled = importlib.import_module('core.finite_strip_numba')
    old_threads = numba.get_num_threads()
    monkeypatch.setattr(dispatch, 'PARALLEL_MIN_CELLS', 1 if parallel else 10**9)
    try:
        numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
        mass, energy, minimum_f, minimum_g, finite = compiled.population_statistics(f, g, kinetic)
    finally:
        numba.set_num_threads(old_threads)
    np.testing.assert_allclose([mass, energy], [f.sum(), np.sum(f*kinetic+g)], rtol=5e-15, atol=1e-12)
    assert minimum_f == f.min() and minimum_g == g.min() and finite
    f[3, 4, 5] = np.nan
    assert compiled.population_statistics(f, g, kinetic)[-1] is False


def test_collector_reuses_accepted_population_statistics_without_skipping_audits(monkeypatch):
    import simulation.finite_strip as lifecycle

    config = prepare_strip_case(dict(name='fused_audit', kind='pulse', dx_m=1e-5,
        width_m=.0004, height_m=.00016, probe_radius_m=.00004,
        pulse_sigma_m=.00003, sponge_width_m=0., sponge_rate=0.))
    run = dict(config['strip'], steps=5, sample_every=2)
    gas = FiniteStripSolver(config, film_width_m=0.)
    original = lifecycle._totals
    calls = []

    def initial_totals_only(solver):
        calls.append(solver.t_lu)
        return original(solver)

    monkeypatch.setattr(lifecycle, '_totals', initial_totals_only)
    data = lifecycle.collect_strip(gas, run)
    assert calls == [0], 'Accepted audited storage must not be recomputed by the collector'
    assert len(data['audit_t_s']) == 5
    f, g = gas._require_state()
    m = gas.mapping
    energy = np.sum(.5*f*np.sum(gas.lattice.c**2, axis=1)+g)*m.pressure_scale*m.lattice.dx_m**2
    np.testing.assert_allclose(data['gas_mass_lu'][-1], f.sum(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(data['gas_energy_J_m'][-1], energy, rtol=0, atol=1e-15)
    assert gas.last_audit.minimum_f == f.min()
    assert gas.last_audit.minimum_g == g.min()
