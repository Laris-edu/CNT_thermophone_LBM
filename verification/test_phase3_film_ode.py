"""P3-3 standalone film ODE fixture tests."""

from __future__ import annotations

import math

import numpy as np

from coupling.drive import ConstantDrive, GaussianPulseDrive, SinusoidalDrive, StepDrive
from coupling.film_ode import (
    FilmOdeParams,
    adiabatic_ramp_solution,
    integrate_film_ode,
    linear_leak_step_solution,
    sinusoidal_steady_temperature,
    sinusoidal_temperature_hat,
)
from phase3_interfaces.complex_amplitude import complex_amplitude


def test_phase3_drive_definitions_match_contract():
    step = StepDrive(power_density_si=1_000.0, t0_si=2.0e-6)
    assert step(1.0e-6) == 0.0
    assert step(2.0e-6) == 1_000.0

    pulse = GaussianPulseDrive(amplitude_si=500.0, t0_si=3.0e-6, sigma_si=1.0e-6)
    assert pulse(3.0e-6) == 500.0
    assert math.isclose(pulse(4.0e-6), 500.0 * math.exp(-1.0), rel_tol=1.0e-15)

    harmonic = SinusoidalDrive(mean_si=10.0, amplitude_hat_si=3.0 - 4.0j, frequency_hz=10_000.0)
    quarter_period = 0.25 / harmonic.frequency_hz
    assert math.isclose(harmonic(0.0), 13.0, rel_tol=0.0, abs_tol=1.0e-14)
    assert math.isclose(harmonic(quarter_period), 14.0, rel_tol=0.0, abs_tol=1.0e-14)


def test_adiabatic_ramp_fixture_is_linear_not_exponential():
    params = FilmOdeParams(C_A_si=7.0e-4, T_ref_K=300.0)
    t = np.linspace(0.0, 2.0e-5, 129)
    drive = ConstantDrive(power_density_si=1_000.0)
    trajectory = integrate_film_ode(
        t_si=t,
        T_initial_K=300.0,
        params=params,
        drive=drive,
        q_g_one_sided_si=0.0,
        method="heun",
    )
    expected = adiabatic_ramp_solution(
        t,
        T_initial_K=300.0,
        P_constant_si=1_000.0,
        C_A_si=params.C_A_si,
    )
    np.testing.assert_allclose(trajectory.T_s_K, expected, rtol=1.0e-14, atol=1.0e-12)
    np.testing.assert_allclose(trajectory.ode_pointwise_residual_si, 0.0, rtol=0.0, atol=1.0e-12)


def test_linear_leak_exponential_fixture_matches_closed_form():
    params = FilmOdeParams(
        C_A_si=7.0e-4,
        T_ref_K=300.0,
        linear_leak_conductance_si=25.0,
    )
    t = np.linspace(0.0, 5.0e-4, 2001)
    trajectory = integrate_film_ode(
        t_si=t,
        T_initial_K=302.0,
        params=params,
        drive=ConstantDrive(power_density_si=1_000.0),
        q_g_one_sided_si=0.0,
        method="rk4",
    )
    expected = linear_leak_step_solution(
        t,
        T_initial_K=302.0,
        P_constant_si=1_000.0,
        params=params,
    )
    np.testing.assert_allclose(trajectory.T_s_K, expected, rtol=5.0e-9, atol=5.0e-9)


def test_sinusoidal_reference_fixture_uses_exp_iomega_t_convention():
    params = FilmOdeParams(
        C_A_si=7.0e-4,
        T_ref_K=300.0,
        linear_leak_conductance_si=25.0,
    )
    frequency_hz = 10_000.0
    P_hat = 1_000.0 - 250.0j
    t = np.arange(512, dtype=float) / (512.0 * frequency_hz)
    expected_hat = sinusoidal_temperature_hat(
        params=params,
        frequency_hz=frequency_hz,
        P_hat_si=P_hat,
    )
    T_s = sinusoidal_steady_temperature(
        t,
        params=params,
        frequency_hz=frequency_hz,
        P_hat_si=P_hat,
    )
    fitted_hat = complex_amplitude(t, T_s - params.T_ref_K, frequency_hz)
    np.testing.assert_allclose(fitted_hat, expected_hat, rtol=1.0e-12, atol=1.0e-12)


def test_rk4_sinusoidal_fixture_keeps_phase_and_amplitude():
    params = FilmOdeParams(
        C_A_si=7.0e-4,
        T_ref_K=300.0,
        linear_leak_conductance_si=25.0,
    )
    frequency_hz = 10_000.0
    P_hat = 800.0 + 150.0j
    expected_hat = sinusoidal_temperature_hat(
        params=params,
        frequency_hz=frequency_hz,
        P_hat_si=P_hat,
    )
    t = np.arange(1024, dtype=float) / (512.0 * frequency_hz)
    trajectory = integrate_film_ode(
        t_si=t,
        T_initial_K=params.T_ref_K + expected_hat.real,
        params=params,
        drive=SinusoidalDrive(mean_si=0.0, amplitude_hat_si=P_hat, frequency_hz=frequency_hz),
        q_g_one_sided_si=0.0,
        method="rk4",
    )
    last_period = t >= (1.0 / frequency_hz)
    fitted_hat = complex_amplitude(
        trajectory.t_si[last_period],
        trajectory.T_s_K[last_period] - params.T_ref_K,
        frequency_hz,
    )
    np.testing.assert_allclose(fitted_hat, expected_hat, rtol=5.0e-7, atol=5.0e-7)


def test_integrated_energy_residual_audit_is_nonvacuous():
    # Contract §7.3 integrated energy balance must pass for an accurate integrator and,
    # crucially, FAIL for a visibly wrong one -- unlike the pointwise rhs residual, which
    # is ~0 by construction regardless of integration quality.
    params = FilmOdeParams(C_A_si=7.0e-4, T_ref_K=300.0, linear_leak_conductance_si=25.0)
    power = 1_000.0

    t_fine = np.linspace(0.0, 5.0e-4, 2001)
    rk4 = integrate_film_ode(
        t_si=t_fine,
        T_initial_K=302.0,
        params=params,
        drive=ConstantDrive(power_density_si=power),
        q_g_one_sided_si=0.0,
        method="rk4",
    )
    scale_fine = power * (t_fine[-1] - t_fine[0])
    assert np.max(np.abs(rk4.energy_residual_si)) / scale_fine < 1.0e-3

    t_coarse = np.linspace(0.0, 5.0e-4, 21)
    euler = integrate_film_ode(
        t_si=t_coarse,
        T_initial_K=302.0,
        params=params,
        drive=ConstantDrive(power_density_si=power),
        q_g_one_sided_si=0.0,
        method="euler",
    )
    scale_coarse = power * (t_coarse[-1] - t_coarse[0])
    # Pointwise residual stays ~0 (vacuous) even though the trajectory is wrong ...
    assert np.max(np.abs(euler.ode_pointwise_residual_si)) < 1.0e-9
    # ... while the integrated audit correctly exceeds the 1% gate.
    assert np.max(np.abs(euler.energy_residual_si)) / scale_coarse > 1.0e-2


def test_euler_integrates_adiabatic_ramp_exactly():
    # Euler is exact for the constant-rhs adiabatic ramp; confirms the euler path is wired.
    params = FilmOdeParams(C_A_si=7.0e-4, T_ref_K=300.0)
    t = np.linspace(0.0, 2.0e-5, 65)
    trajectory = integrate_film_ode(
        t_si=t,
        T_initial_K=300.0,
        params=params,
        drive=ConstantDrive(power_density_si=1_000.0),
        q_g_one_sided_si=0.0,
        method="euler",
    )
    expected = adiabatic_ramp_solution(
        t,
        T_initial_K=300.0,
        P_constant_si=1_000.0,
        C_A_si=params.C_A_si,
    )
    # Euler is analytically exact here; the only gap is rounding from sequential
    # accumulation vs the closed-form single multiply (~1e-12 K after 64 steps).
    np.testing.assert_allclose(trajectory.T_s_K, expected, rtol=1.0e-12, atol=1.0e-9)
