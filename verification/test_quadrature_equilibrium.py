"""独立验证求积平衡态的分布结构、矩与正性；不沿用旧配矩答案。"""
import math

import numpy as np
import pytest

from core.equilibrium import equilibrium_fg, feq_hermite4, geq_polyatomic, raw_moments
from core.lattice import make_lattice
from core.macroscopic import recover_macro


def normal_moment(n, mean, variance):
    return sum(math.comb(n, 2*j)*math.factorial(2*j)/(2**j*math.factorial(j))
               *variance**j*mean**(n-2*j) for j in range(n//2+1))


def hermite(n, x, variance):
    return sum(math.factorial(n)/math.factorial(n-2*j)/math.factorial(j)
               *(-variance/2)**j*x**(n-2*j) for j in range(n//2+1))


def tensor_product_reference(rho, u, theta, lattice, order):
    """独立多重指标构造，避免逐项照抄实现中的旋转不变量公式。"""
    q=lattice.theta_q
    out=np.zeros(lattice.q)
    for total in range(order+1):
        for m in range(total+1):
            n=total-m
            coeff=normal_moment(m,u[0],theta-q)*normal_moment(n,u[1],theta-q)
            out += coeff*hermite(m,lattice.c[:,0],q)*hermite(n,lattice.c[:,1],q)/(math.factorial(m)*math.factorial(n)*q**total)
    return rho*lattice.w*out


def test_reference_equilibrium_is_the_quadrature_measure_through_eighth_order():
    l=make_lattice('D2Q37');rho=1.3;t=l.theta_q
    f,g=equilibrium_fg(rho,np.zeros(2),t,3,l)
    np.testing.assert_allclose(f,rho*l.w,rtol=2e-14,atol=2e-16)
    np.testing.assert_allclose(g,1.5*rho*t*l.w,rtol=2e-14,atol=2e-16)
    for (m,n),v in raw_moments(f,l,8).items():
        assert v==pytest.approx(rho*normal_moment(m,0.,t)*normal_moment(n,0.,t),abs=2e-12)
    assert f.min()>0 and g.min()>0


@pytest.mark.parametrize('order',[2,3,4])
@pytest.mark.parametrize('ratio',[.9,1.,1.2])
def test_matches_independent_weighted_tensor_expansion(order,ratio):
    l=make_lattice('D2Q37');rho=.93;t=ratio*l.theta_q;u=np.array([.025,-.013])
    g=geq_polyatomic(rho,u,t,3,l,order=order)
    expected=tensor_product_reference(1.5*rho*t,u,t,l,order)
    np.testing.assert_allclose(g,expected,rtol=2e-13,atol=2e-16)
    f=feq_hermite4(rho,u,t,l)
    np.testing.assert_allclose(f,tensor_product_reference(rho,u,t,l,4),rtol=2e-13,atol=2e-16)
    for (m,n),v in raw_moments(g,l,order).items():
        expected=1.5*rho*t*normal_moment(m,u[0],t)*normal_moment(n,u[1],t)
        assert v==pytest.approx(expected,rel=2e-12,abs=2e-13)
    for (m,n),v in raw_moments(f,l,4).items():
        assert v==pytest.approx(rho*normal_moment(m,u[0],t)*normal_moment(n,u[1],t),rel=2e-12,abs=2e-13)


def test_project_temperature_mach_grid_positive_and_recovers_macro():
    l=make_lattice('D2Q37')
    ratio,mach,angle=np.meshgrid(np.linspace(.9,1.2,61),np.linspace(0,.05,6),np.arange(16)*np.pi/8,indexing='ij')
    t=ratio*l.theta_q
    u=(mach*np.sqrt(1.4*t))[...,None]*np.stack([np.cos(angle),np.sin(angle)],axis=-1)
    f,g=equilibrium_fg(np.ones_like(t),u,t,3,l)
    assert f.min()>0 and g.min()>0
    m=recover_macro(f,g,D=2,S=3,lattice=l)
    np.testing.assert_allclose(m.rho,1.,atol=3e-14)
    np.testing.assert_allclose(m.u,u,atol=3e-14)
    np.testing.assert_allclose(m.theta,t,atol=3e-14)


def test_retired_d2q21_velocity_set_is_rejected_loudly():
    # D2Q21 已退役：四阶平衡态需要八阶求积。旧配置必须立即报错，
    # 不能静默退化成不能运行的速度集。
    with pytest.raises(ValueError,match='D2Q21|D2Q37'):
        make_lattice('D2Q21')


@pytest.mark.parametrize('theta,u',[(.048,np.zeros(2)),(.7,np.array([2.,0.])),(-1.,np.zeros(2)),(float('nan'),np.zeros(2))])
def test_unsafe_equilibrium_states_are_rejected(theta,u):
    with pytest.raises(ValueError,match='equilibrium|temperature|finite|positive'):
        equilibrium_fg(1.,u,theta,3,make_lattice('D2Q37'))


def test_internal_degrees_zero_is_zero_distribution_and_negative_is_rejected():
    l=make_lattice('D2Q37')
    np.testing.assert_array_equal(geq_polyatomic(1.,np.zeros(2),l.theta_q,0,l),np.zeros(l.q))
    with pytest.raises(ValueError,match='S|degrees'):
        geq_polyatomic(1.,np.zeros(2),l.theta_q,-1,l)


def test_diagnostic_domain_probe_does_not_clip_negative_populations():
    l=make_lattice('D2Q37');theta=.048
    f=feq_hermite4(1.,np.zeros(2),theta,l,check_positive=False)
    assert f.min()<0
    np.testing.assert_allclose(f,tensor_product_reference(1.,np.zeros(2),theta,l,4),rtol=3e-13,atol=2e-16)
    assert np.sum(f)==pytest.approx(1.,abs=3e-14)


def test_broadcast_density_temperature_velocity_shapes():
    l=make_lattice('D2Q37')
    rho=np.array([.7,1.2])[:,None];theta=l.theta_q*np.array([.9,1.,1.2])[None,:]
    u=np.array([.013,-.008])
    f,g=equilibrium_fg(rho,u,theta,3,l)
    assert f.shape==g.shape==(2,3,37)
    m=recover_macro(f,g,lattice=l)
    np.testing.assert_allclose(m.rho,np.broadcast_to(rho,(2,3)),atol=3e-14)
    np.testing.assert_allclose(m.theta,np.broadcast_to(theta,(2,3)),atol=3e-14)
    with pytest.raises(ValueError,match='density'):
        geq_polyatomic(-1.,u,l.theta_q,0,l)


def test_off_reference_higher_moments_are_truncation_errors_not_exact_maxwellian():
    l=make_lattice('D2Q37');t=1.1*l.theta_q
    f,g=equilibrium_fg(1.,np.zeros(2),t,3,l)
    fm=raw_moments(f,l,6);gm=raw_moments(g,l,4)
    assert fm[(4,0)]==pytest.approx(3*t*t,rel=2e-13)
    assert gm[(2,0)]==pytest.approx(1.5*t*t,rel=2e-13)
    # 目标阶数外的矩不应被报告为任意温度下的 Maxwellian 恒等式。
    assert not np.isclose(fm[(6,0)],15*t**3,rtol=1e-4,atol=0)
    assert not np.isclose(gm[(4,0)],4.5*t**3,rtol=1e-4,atol=0)
