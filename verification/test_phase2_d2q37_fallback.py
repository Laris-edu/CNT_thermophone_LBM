import numpy as np

from core.lattice_d2q37 import assert_d2q37_moments, make_d2q37, moment


def test_d2q37_candidate_has_positive_weights_and_opposites():
    lattice = make_d2q37()
    assert lattice.c.shape == (37, 2)
    assert lattice.w.shape == (37,)
    assert lattice.q == 37
    assert lattice.d == 2
    assert np.all(lattice.w > 0.0)
    assert np.isclose(np.sum(lattice.w), 1.0)
    for index, opposite in enumerate(lattice.opposite):
        assert np.allclose(lattice.c[index] + lattice.c[opposite], 0.0)


def test_d2q37_candidate_matches_eighth_order_even_moments():
    assert_d2q37_moments(tol=1.0e-12)
    lattice = make_d2q37()
    assert np.isclose(moment((2, 0), lattice), lattice.theta_q)
    assert np.isclose(moment((8, 0), lattice), 105.0 * lattice.theta_q**4)
    assert np.isclose(moment((4, 4), lattice), 9.0 * lattice.theta_q**4)
