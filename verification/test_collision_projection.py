"""加速投影与原逐格中心矩 Gram 求解在有限流速下等价。"""
import numpy as np
import pytest
import core.collision_smrt as collision
from core.lattice import make_lattice


@pytest.mark.parametrize('kind,size,start', [('first',3,1),('second',6,3),('heat',8,6)])
def test_cached_projection_matches_full_moving_basis(kind,size,start):
    lattice=make_lattice()
    rng=np.random.default_rng(2049)
    u=rng.uniform(-.3,.3,size=(3,7,2))
    delta=np.zeros((3,7,size))
    delta[...,start:]=rng.normal(size=(3,7,size-start))
    basis_function=getattr(collision, '_central_'+{'first':'first_moment_matrix',
        'second':'second_moment_matrix','heat':'heat_flux_moment_matrix'}[kind])
    project_function=getattr(collision, '_weighted_minimum_norm_'+{'first':'first_order_delta',
        'second':'second_order_delta','heat':'heat_flux_delta'}[kind])
    basis=basis_function(lattice,u)
    gram=np.einsum('...ia,...ja->...ij',basis*lattice.w,basis)
    multipliers=np.linalg.solve(gram,delta[...,None])[...,0]
    expected=np.einsum('...ia,...i,a->...a',basis,multipliers,lattice.w)
    actual=project_function(delta,u,lattice)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(np.einsum('...ia,...a->...i',basis,actual),delta,rtol=2e-13,atol=2e-14)
