"""Numerical propagation gate; fabricated arrays test decisions, not physics."""
import numpy as np
import pytest

from Compute.finite_strip.chain import validate_propagation
from farfield.strip_surface import upper_rectangle,mirror_strip_surface
from scipy.special import hankel2


def fixture():
    frequency,rho,c=20000.,1.16,347.
    upper=upper_rectangle(.0007,.0002,5e-6)
    k=2*np.pi*frequency/c
    r=np.hypot(upper['x_m'],upper['y_m'])
    p=hankel2(0,k*r)*np.exp(.4j)
    ur=-1j*hankel2(1,k*r)*np.exp(.4j)/(rho*c)
    closed=mirror_strip_surface(upper,p,ur*upper['x_m']/r,ur*upper['y_m']/r)
    ox,oy=np.array([0.,0.,0.]),np.array([.002,.045,0.])
    hat=hankel2(0,k*oy[:2])*np.exp(.4j)
    hat=np.r_[hat,0j]
    t=np.arange(1,129)/(64*frequency)
    source=dict(**closed,frequency_Hz=frequency,rho0_kg_m3=rho,c0_m_s=c,
        source_validation_passed=1,observer_x_m=ox,observer_y_m=oy,
        expected_grid_observer_x_m=ox,expected_grid_observer_y_m=oy,
        observer_role=np.array(['nearfield','farfield_45mm','interior_cancel']))
    result=dict(observer_x_m=ox,observer_y_m=oy,observer_p_hat_Pa=hat,
                time_s=t,observer_pressure_Pa=np.real(hat[:,None]*np.exp(2j*np.pi*frequency*t)))
    return source,result


def test_propagation_gate_uses_actual_coordinates_and_absolute_phase():
    source,result=fixture()
    check=validate_propagation(source,result)
    assert check['propagation_validation_passed']
    result['observer_p_hat_Pa']=result['observer_p_hat_Pa']*np.exp(.03j)
    check=validate_propagation(source,result)
    assert not check['propagation_validation_passed']
    assert check['max_phase_error_deg']>1.


def test_interior_leakage_is_a_separate_gate():
    source,result=fixture()
    result['observer_p_hat_Pa'][-1]=.02*max(abs(result['observer_p_hat_Pa']))
    check=validate_propagation(source,result)
    assert check['max_amplitude_error_relative']<.01
    assert check['interior_leakage_relative']>.01
    assert not check['propagation_validation_passed']


def test_coordinate_mismatch_is_not_compared_as_if_same_position():
    source,result=fixture()
    result['observer_y_m']=result['observer_y_m']+.00005
    with pytest.raises(ValueError,match='coordinates'):
        validate_propagation(source,result)


def test_saved_time_series_must_match_reported_complex_pressure():
    source,result=fixture()
    result['observer_pressure_Pa']*=2
    check=validate_propagation(source,result)
    assert not check['propagation_validation_passed']
