import math

import numpy as np

from reference.continuum_1d_freq import solve_level_A_frequency, solve_level_B_frequency
from reference.constants import relative_error
from reference.thermal_admittance import thermal_admittance_halfspace


def _phase_error_deg(value, expected):
    return abs(math.degrees(np.angle(value / expected)))


def test_level_A_halfspace_admittance():
    for f_hz in [1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0]:
        result = solve_level_A_frequency(f_hz, 1.0 + 0.0j)
        exact = thermal_admittance_halfspace(f_hz)
        assert relative_error(result.q_g_hat, exact) < 1.0e-2
        assert _phase_error_deg(result.q_g_hat, exact) < 1.0


def test_level_B_halfspace_temperature():
    q_hat = 1.0 + 0.0j
    for f_hz in [1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0]:
        result = solve_level_B_frequency(f_hz, q_hat)
        exact = q_hat / thermal_admittance_halfspace(f_hz)
        assert relative_error(result.T_s_hat, exact) < 1.0e-2
        assert _phase_error_deg(result.T_s_hat, exact) < 1.0

