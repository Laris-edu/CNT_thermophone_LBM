import numpy as np

from core.finite_strip_solver import FiniteStripSolver
from verification.test_finite_strip import solver


def test_reinitializing_strip_resets_pml_history_and_previous_audit():
    original=solver()
    gas=FiniteStripSolver(original.config,film_width_m=8*original.film_dx_m,pml_cells=5)
    gas.step_temperature(300.01)
    assert gas.last_audit is not None
    assert np.max(abs(gas.pml.memory))>0
    gas.initialize_from_macro(gas.mapping.lattice.rho_ref_lu,np.zeros(2),gas.mapping.theta_ref_lu)
    assert gas.t_lu==0
    assert gas.last_audit is None
    np.testing.assert_array_equal(gas.pml.memory,0)
    gas.step_temperature(300.)
    np.testing.assert_allclose(gas.get_macro().u,0,atol=2e-14)
