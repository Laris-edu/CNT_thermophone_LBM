from reference.continuum_1d_freq import solve_level_C_frequency
from reference.constants import default_params, relative_error
from reference.film_ode import level_c_closed_form


def test_level_C_closed_form_energy_residual():
    params = default_params()
    closed = level_c_closed_form(10_000.0, 1000.0, C_A=params.C_A, params=params)

    assert closed.energy_residual_rel < 1.0e-10


def test_level_C_frequency_matches_closed_form():
    params = default_params()
    closed = level_c_closed_form(10_000.0, 1000.0, C_A=params.C_A, params=params)
    result = solve_level_C_frequency(10_000.0, 1000.0, params.C_A, params=params)

    assert relative_error(result.T_s_hat, closed.T_s_hat) < 1.0e-12
    assert relative_error(result.q_g_hat, closed.q_g_hat) < 1.0e-12
    assert result.energy_residual_rel < 1.0e-3

