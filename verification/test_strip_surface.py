import numpy as np
from scipy.special import hankel2
from farfield.strip_surface import upper_rectangle, mirror_strip_surface
from farfield.kirchhoff_2d import kirchhoff_2d_frequency, dpdn_from_velocity, GREEN_OUTGOING


def test_mirrored_three_sides_reconstruct_cylindrical_exterior_without_double_gain():
    surface=upper_rectangle(.0007,.0002,2e-6)
    x,y=surface['x_m'],surface['y_m']; r=np.hypot(x,y)
    k=2*np.pi*20000/347; rho=1.16; omega=k*347
    p=hankel2(0,k*r)
    derivative=-k*hankel2(1,k*r)
    ux=derivative*x/r/(-1j*omega*rho)
    uy=derivative*y/r/(-1j*omega*rho)
    complete=mirror_strip_surface(surface,p,ux,uy)
    assert abs(np.sum(complete['normal_x']*complete['ds_m']))<1e-15
    assert abs(np.sum(complete['normal_y']*complete['ds_m']))<1e-15
    out=kirchhoff_2d_frequency(surface_x_m=complete['x_m'],surface_y_m=complete['y_m'],
        normal_x=complete['normal_x'],normal_y=complete['normal_y'],ds_m=complete['ds_m'],
        p_hat_Pa=complete['p_hat_Pa'],dpdn_hat_Pa_m=dpdn_from_velocity(complete['un_hat_m_s'],
        omega_rad_s=omega,rho0_kg_m3=rho),observer_x_m=np.array([0.,.001]),
        observer_y_m=np.array([.045,.002]),omega_rad_s=omega,c0_m_s=347.,green_convention=GREEN_OUTGOING)
    expected=hankel2(0,k*np.hypot([0.,.001],[.045,.002]))
    np.testing.assert_allclose(out,expected,rtol=2e-5)
