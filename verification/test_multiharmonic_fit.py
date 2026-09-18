"""WP1-1 multiharmonic joint fitter tests (contract Phase5_instruct_v1.2 §12.1/§12.2).

Instrument certification for ``postproc/multiharmonic_fit.py``: frozen phase
convention against the project single source, in-basis recovery with a
non-target leakage floor, the pre-registered detrend registry, trend
absorption with a wrong-detrend counterexample, fit-only U95 against seeded
noise, residual spectrum on out-of-basis contamination, multi-window
sensitivity with a drifting counterexample, and input validation.

These tests certify the WP1 instrument only — they do not claim any Phase_5
physics gate (all gates remain tracked in docs/Phase_5/Phase5_STATUS.md).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from phase3_interfaces.complex_amplitude import complex_amplitude
from postproc.multiharmonic_fit import (
    PROTOCOL_DETREND,
    fit_multiharmonic,
    leakage_fixture,
    multiwindow_sensitivity,
    probe_amplitude,
    residual_spectrum_fft,
    suffix_cycle_windows,
)

F0_HZ = 10_000.0
OMEGA = 2.0 * math.pi * F0_HZ


def _grid(n_cycles: int = 8, samples_per_cycle: int = 64) -> np.ndarray:
    """Uniform endpoint-excluded grid (DFT-orthogonal for integer harmonics)."""

    dt = 1.0 / (F0_HZ * samples_per_cycle)
    return np.arange(n_cycles * samples_per_cycle, dtype=float) * dt


def _tone(t: np.ndarray, n: int, x_hat: complex) -> np.ndarray:
    """Independent synthesis (plain cos/sin, no module helper): Re[x_hat e^{+inΩt}]."""

    return x_hat.real * np.cos(n * OMEGA * t) - x_hat.imag * np.sin(n * OMEGA * t)


def test_phase_convention_matches_frozen_single_source():
    t = _grid()
    a_cos, b_sin, dc = 0.31, -0.22, 0.07
    x = dc + a_cos * np.cos(OMEGA * t) + b_sin * np.sin(OMEGA * t)

    fit = fit_multiharmonic(t, x, OMEGA, n_harmonics=5, detrend_order=0)

    expected = complex(a_cos, 0.0) - 1j * b_sin
    assert abs(fit.harmonic(1) - expected) < 1.0e-12
    # cross-check against the frozen single source (DC orthogonal on this grid)
    assert abs(fit.harmonic(1) - complex_amplitude(t, x, F0_HZ)) < 1.0e-12
    # convention closure: x(t) = Re[x_hat exp(+i Omega t)] reconstructs the record
    recon = dc + np.real(fit.harmonic(1) * np.exp(1j * OMEGA * t))
    assert float(np.max(np.abs(recon - x))) < 1.0e-12


def test_in_basis_recovery_and_nontarget_floor():
    t = _grid()
    truth = {
        1: 0.37269 * np.exp(-1j * math.radians(25.0)),
        2: 0.010 * np.exp(1j * math.radians(40.0)),
        3: 1.0e-3 * np.exp(1j * math.radians(170.0)),
    }
    x = 0.05 + sum(_tone(t, n, x_hat) for n, x_hat in truth.items())

    fit = fit_multiharmonic(t, x, OMEGA, n_harmonics=5, detrend_order=0)

    for n, x_hat in truth.items():
        assert abs(fit.harmonic(n) - x_hat) / abs(x_hat) < 1.0e-10
    for n in (4, 5):
        assert fit.amplitude(n) / fit.amplitude(1) < 1.0e-8
    assert abs(fit.trend_coeffs_raw[0] - 0.05) < 1.0e-12


def test_leakage_fixture_meets_contract_floor():
    result = leakage_fixture(OMEGA)
    assert result["passed"] is True
    assert result["max_relative_leakage"] <= 1.0e-8
    assert result["threshold"] == 1.0e-8
    assert result["target_relative_error"] < 1.0e-10
    assert result["fit_window"]["n_samples"] == 8 * 64


def test_leakage_metric_detects_real_contamination():
    # non-degeneracy: the metric must be able to fail on genuine 2f content
    t = _grid()
    x = 1.0 * np.cos(OMEGA * t) + 1.0e-4 * np.cos(2.0 * OMEGA * t)
    fit = fit_multiharmonic(t, x, OMEGA)
    assert fit.leakage_relative(target=1)[2] > 1.0e-5


def test_trend_absorption_and_wrong_detrend_counterexample():
    t = _grid()
    x_hat_1 = 0.37269 * np.exp(-1j * math.radians(25.0))
    a0, a1, a2 = 0.03, -40.0, 3.125e5  # quadratic drift reaches ~0.2 over the window
    x = a0 + a1 * t + a2 * t**2 + _tone(t, 1, x_hat_1)

    good = fit_multiharmonic(t, x, OMEGA, detrend_order=2)
    assert abs(good.harmonic(1) - x_hat_1) / abs(x_hat_1) < 1.0e-9
    assert np.allclose(good.trend_coeffs_raw, [a0, a1, a2], rtol=1.0e-7, atol=1.0e-9)

    # counterexample: pre-registered knob is load-bearing — omitting the trend
    # (detrend_order=0) must visibly bias the fundamental
    bad = fit_multiharmonic(t, x, OMEGA, detrend_order=0)
    assert abs(bad.harmonic(1) - x_hat_1) / abs(x_hat_1) > 1.0e-3


def test_noise_sigma_and_u95_fit():
    t = _grid()
    x_hat_1 = 0.37269 * np.exp(-1j * math.radians(25.0))
    sigma_true = 1.0e-3
    rng = np.random.default_rng(20260721)
    x = _tone(t, 1, x_hat_1) + rng.normal(0.0, sigma_true, t.size)

    fit = fit_multiharmonic(t, x, OMEGA, n_harmonics=5, detrend_order=0)

    assert fit.dof == t.size - 11
    assert 0.85 * sigma_true < fit.noise_sigma < 1.15 * sigma_true
    sigma_ab = sigma_true * math.sqrt(2.0 / t.size)  # per-coefficient noise scale
    assert 0.5 < fit.amplitude_u95(1) / (1.96 * sigma_ab) < 2.0
    assert abs(fit.harmonic(1) - x_hat_1) < 5.0 * sigma_ab
    assert fit.phase_u95_deg(1) > 0.0
    assert np.isfinite(fit.condition_number) and fit.condition_number < 50.0


def test_residual_spectrum_sees_out_of_basis_tone():
    t = _grid()
    amp_7f = 1.0e-3
    x = _tone(t, 1, 0.37269 + 0.0j) + amp_7f * np.cos(7.0 * OMEGA * t)

    fit = fit_multiharmonic(t, x, OMEGA, n_harmonics=5)

    # in-basis harmonics stay clean (7f is DFT-orthogonal on this grid)
    assert max(fit.leakage_relative(target=1).values()) < 1.0e-8
    freq_hz, amp = residual_spectrum_fft(fit)
    k_peak = int(np.argmax(amp))
    assert abs(freq_hz[k_peak] - 7.0 * F0_HZ) < 1.0e-6
    assert abs(amp[k_peak] - amp_7f) / amp_7f < 1.0e-2
    probe = probe_amplitude(fit.t, fit.residual, 7.0 * F0_HZ)
    assert abs(abs(probe) - amp_7f) / amp_7f < 1.0e-6


def test_protocol_detrend_registry_frozen():
    # transcription guard for contract §12.2 (must not drift silently)
    assert PROTOCOL_DETREND == {
        "A1": (0, 1),
        "A2a": (0, 1),
        "A2b": (1, 2),
        "G2_linear_fixture": (0, None),
    }


def test_multiwindow_sensitivity_stationary_and_drift_counterexample():
    t = _grid(n_cycles=12)
    x_hat_1 = 0.5 * np.exp(1j * math.radians(10.0))
    windows = suffix_cycle_windows(t, OMEGA, [4, 6, 8])

    stationary = _tone(t, 1, x_hat_1)
    sens = multiwindow_sensitivity(
        t, stationary, OMEGA, n_harmonics=3, detrend_order=0, windows=windows
    )
    assert sens.amplitude_max_rel_dev[1] < 1.0e-9
    assert sens.phase_max_dev_deg[1] < 1.0e-6

    # counterexample: a 20% amplitude ramp must register as window sensitivity
    drifting = (1.0 + 0.2 * t / t[-1]) * stationary
    sens_drift = multiwindow_sensitivity(
        t, drifting, OMEGA, n_harmonics=3, detrend_order=0, windows=windows
    )
    assert sens_drift.amplitude_max_rel_dev[1] > 1.0e-2


def test_json_payload_carries_contract_fields():
    t = _grid()
    fit = fit_multiharmonic(t, _tone(t, 1, 0.2 - 0.1j), OMEGA)
    payload = fit.to_json_payload()
    for key in (
        "phase_convention",
        "harmonic_order_max",
        "detrend_order",
        "fit_window",
        "fit_cycles",
        "harmonic_fit_condition_number",
        "amplitude_u95_fit",
        "phase_u95_fit_deg",
        "residual_rms",
    ):
        assert key in payload
    json.dumps(payload)  # must be serializable for harmonic_fit.json (contract §16.1)


def test_input_validation_and_uneven_sampling():
    t = _grid()
    x = np.cos(OMEGA * t)

    with pytest.raises(ValueError):
        fit_multiharmonic(t, x, OMEGA, detrend_order=3)
    with pytest.raises(ValueError):
        fit_multiharmonic(t[:8], x[:8], OMEGA)  # fewer samples than parameters
    with pytest.raises(ValueError):
        fit_multiharmonic(t, x, -OMEGA)
    bad = x.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        fit_multiharmonic(t, bad, OMEGA)

    # uneven sampling: LS fit still exact for in-basis signals; FFT path refuses
    rng = np.random.default_rng(7)
    dt = t[1] - t[0]
    jitter = np.concatenate(([0.0], rng.uniform(-0.3, 0.3, t.size - 2) * dt, [0.0]))
    t_jitter = t + jitter
    assert np.all(np.diff(t_jitter) > 0.0)
    x_hat = 0.4 - 0.1j
    x_jitter = x_hat.real * np.cos(OMEGA * t_jitter) - x_hat.imag * np.sin(OMEGA * t_jitter)
    fit = fit_multiharmonic(t_jitter, x_jitter, OMEGA)
    assert abs(fit.harmonic(1) - x_hat) < 1.0e-9
    with pytest.raises(ValueError):
        residual_spectrum_fft(fit)
