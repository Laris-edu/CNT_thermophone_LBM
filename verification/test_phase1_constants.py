import math

from reference.constants import default_params, thermal_scales


def _rel(value, expected):
    return abs(value - expected) / abs(expected)


def test_phase1_default_constants_and_scales():
    params = default_params()
    scales = thermal_scales(10_000.0, params, P_hat=1000.0, C_A=params.C_A)

    assert _rel(params.alpha0, 2.223e-5) < 5.0e-3
    assert _rel(scales["delta_T"], 26.6e-6) < 5.0e-3
    assert _rel(scales["delta_v"], 22.4e-6) < 5.0e-3
    assert _rel(params.Pr, 0.706) < 5.0e-3
    assert _rel(scales["Pi_C"], 2.22e-2) < 5.0e-3
    assert _rel(scales["epsilon_P"], 1.69e-3) < 5.0e-3
    assert math.isclose(params.beta0, 0.0)

