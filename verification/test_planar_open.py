"""开放域的链接覆盖、独立边界收支及脉冲反射回归。"""
import numpy as np
import pytest

from core.lattice import make_lattice
from core.streaming import pull_stream_open_y
from simulation.planar import make_solver, pulse_case


def test_open_streaming_has_no_y_wrap_and_covers_every_multispeed_link():
    lattice = make_lattice()
    f = np.arange(9*4*lattice.q,dtype=float).reshape(9,4,lattice.q)
    out = pull_stream_open_y(f,lattice=lattice)
    for y in range(9):
        for x in range(4):
            for a,(cx,cy) in enumerate(lattice.c.astype(int)):
                if 0 <= y-cy < 9:
                    assert out[y,x,a] == f[y-cy,(x-cx)%4,a]
                else:
                    assert np.isnan(out[y,x,a])


def test_open_equilibrium_does_not_generate_pressure_or_heat():
    solver = make_solver(length=50e-6)
    initial = solver.get_macro()
    for _ in range(60):
        audit=solver.step_temperature(300.)
        assert abs(audit.wall_mass_lu)<1e-14
        assert abs(audit.wall_energy_J_m2)<1e-13
    state=solver.get_macro()
    np.testing.assert_allclose(state.p,initial.p,rtol=0,atol=5e-14)
    np.testing.assert_allclose(state.u,0,rtol=0,atol=5e-14)


def test_heated_wall_exchanges_energy_without_mass_and_outlet_is_counted():
    solver=make_solver(length=100e-6)
    old=solver.get_macro()
    m=solver.mapping
    for _ in range(60):
        a=solver.step_temperature(300.1)
        new=solver.get_macro()
        dm=np.sum(new.rho)-np.sum(old.rho)
        de=np.sum(new.E_tot-old.E_tot)*m.pressure_scale*m.lattice.dx_m
        assert abs(a.wall_mass_lu)<1e-14
        assert abs(dm-a.top_mass_lu)<5e-13
        assert abs(de-a.wall_energy_J_m2-a.top_energy_J_m2)<5e-13
        old=new
    assert np.max(new.p)*m.pressure_scale > m.lattice.rho_ref_lu*m.theta_ref_lu*m.pressure_scale


def test_outgoing_pulse_reflection_below_one_percent(tmp_path):
    result=pulse_case(make_solver(),tmp_path)
    assert result['reflection_ratio']<.01


def test_planar_solver_rejects_periodic_step():
    with pytest.raises(TypeError,match='step_temperature'):
        make_solver().step()
