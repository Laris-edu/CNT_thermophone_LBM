"""P4-4 gate K0: the 2D Kirchhoff kernel passes its manufactured fixtures (contract section 9).

Anchors farfield/kirchhoff_2d.py -- the Green convention (outgoing (-i/4) H0^(2)(kR) under the
frozen exp(+i Omega t)), the prefactor, the normal convention and the velocity input channel --
against manufactured Helmholtz fields. Contract kernel-level gates: amplitude error < 2%,
phase error < 2 deg (they bind the KERNEL only; M4 end-to-end stays <10%). The convention is
pinned HERE once and never re-tuned against end-to-end thermoacoustic results.

Reuses the fixture builders of scripts/phase4_kirchhoff_verification.py driven by
configs/phase4_kirchhoff_fixture.yaml. Measured margins (2026-07-09): cylindrical 0.08%/0.03
deg; plane/velocity 0.49%/0.50 deg; under-sampled 1.5/lambda errs at 164% and refinement
converges to the aperture-truncation floor ~0.06%; the hankel1 kernel (an INCOMING wave under
this time convention) fails reconstruction at 104% -- the fixtures can actually SEE a wrong
convention (discriminating power, Phase_3 non-degeneracy lesson)."""

from __future__ import annotations

from pathlib import Path

from config_io import load_config
from scripts.verify_kirchhoff import (
    convergence_passes,
    run_convergence,
    run_counterexample,
    run_cylindrical,
    run_plane_wave,
)

FIXTURE_CONFIG = Path("configs/phase4_kirchhoff_fixture.yaml")


def _cfg():
    return load_config(FIXTURE_CONFIG)


def test_k0_cylindrical_manufactured_field():
    """Fixture 1 (+4a prefactor lock): outgoing cylindrical wave reconstructed at all
    observers within the contract gates -- pins integral + H0^(2) kind + (-i/4) prefactor."""

    cfg = _cfg()
    r = run_cylindrical(cfg)
    assert r["amp_err_max"] < float(cfg["gates"]["amplitude_rel_max"])
    assert r["phase_err_deg_max"] < float(cfg["gates"]["phase_deg_max"])


def test_k0_discrete_convergence():
    """Fixture 2: refinement reduces the error; the finest sampling sits inside the gate and
    the under-sampled end is BADLY wrong (the fixture is sensitive to discretization)."""

    cfg = _cfg()
    series = run_convergence(cfg)["series"]
    assert convergence_passes(
        series,
        float(cfg["gates"]["amplitude_rel_max"]),
        float(cfg["gates"]["phase_deg_max"]),
    )
    assert series[0]["amp_err_max"] > 0.5          # 1.5 samples/lambda must fail hard
    assert series[-1]["amp_err_max"] < float(cfg["gates"]["amplitude_rel_max"])


def test_k0_convergence_rejects_a_refinement_regression_above_the_gate():
    series = [
        {"samples_per_wavelength": 1.0, "amp_err_max": 1.0, "phase_err_deg_max": 20.0},
        {"samples_per_wavelength": 2.0, "amp_err_max": 0.01, "phase_err_deg_max": 1.0},
        {"samples_per_wavelength": 4.0, "amp_err_max": 0.2, "phase_err_deg_max": 8.0},
    ]
    assert not convergence_passes(series, 0.02, 2.0)


def test_k0_phase_convention_velocity_channel():
    """Fixture 3: plane wave fed via dp/dn = -i Omega rho0 v_n transports phase as -k h --
    certifies the exp(+i Omega t) <-> H^(2) pairing on the channel the D3-4 chain uses."""

    cfg = _cfg()
    r = run_plane_wave(cfg)
    assert r["amp_err"] < float(cfg["gates"]["amplitude_rel_max"])
    assert r["phase_err_deg"] < float(cfg["gates"]["phase_deg_max"])


def test_k0_wrong_convention_counterexample():
    """Fixture 4b (non-degeneracy): the hankel1 kernel under exp(+i Omega t) is an incoming
    wave and must FAIL reconstruction by O(1) -- the gates have discriminating power."""

    cfg = _cfg()
    r = run_counterexample(cfg)
    assert r["reconstruction_error"] > float(cfg["gates"]["counterexample_min_error"])
