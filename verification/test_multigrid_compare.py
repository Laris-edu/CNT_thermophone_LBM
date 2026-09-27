"""Saved-data comparisons use aligned physical times and volume-weighted means."""
import numpy as np
import pytest

from Compute.multi_grid.compare import harmonic_comparison
from config_io import load_config
from simulation.study import BASE


def _synthetic_pair():
    config=load_config(BASE)
    config['run']=dict(mode='power',frequency_Hz=1.,cycles=4.,film=dict(T_ref_K=300.))
    t=np.arange(1,257)/64
    film_t=np.arange(1,513)/128
    heat_t=(np.arange(512)+.5)/128
    profile_t=np.array([0.,1.,2.,2.5,3.,3.5,4.])
    def data(z,width,offset):
        temperature=300.+profile_t[:,None]+z[None,:]+offset
        temperature[:3]+=100.  # Early startup must not enter the requested mean.
        return dict(t_s=t.copy(),film_t_s=film_t.copy(),heat_time_s=heat_t.copy(),
            profile_t_s=profile_t.copy(),z_m=z,cell_width_m=width,
            pressure_Pa=(100000+np.sin(2*np.pi*t))[:,None],
            wall_flux_W_m2=10+np.sin(2*np.pi*heat_t),
            film_temperature_K=300.5+.2*film_t+.1*np.sin(2*np.pi*film_t)+offset,
            profile_temperature_K=temperature)
    a=data(np.array([.5,1.5,3.]),np.array([1.,1.,2.]),.2)
    b=data(np.array([.5,1.5,2.5,3.5]),np.ones(4),0.)
    return a,b,config


def test_temperature_means_select_actual_right_endpoints_and_weight_cell_volumes():
    a,b,config=_synthetic_pair()
    result=harmonic_comparison(a,b,config)
    gas=result['gas_volume_mean_temperature_rise_last_two_cycles_K']
    assert gas['uniform']==pytest.approx(5.25)
    assert gas['two_level']==pytest.approx(5.45)
    assert gas['two_level_minus_uniform']==pytest.approx(.2)
    film=result['film_mean_temperature_rise_last_two_cycles_K']
    assert film['uniform']==pytest.approx(1.10078125)
    assert film['two_level']==pytest.approx(1.30078125)
    assert film['two_level_minus_uniform']==pytest.approx(.2)
    window=result['mean_temperature_window']
    assert window['endpoint_rule']=='start < t <= end'
    assert window['profile_samples']==4
    assert window['profile_first_s']==2.5
    assert window['profile_last_s']==4.
    assert result['mean_gas_temperature_max_difference_K']==pytest.approx(.2)


@pytest.mark.parametrize('axis',['heat_time_s','film_t_s','profile_t_s'])
def test_harmonic_comparison_rejects_misaligned_observation_axes(axis):
    a,b,config=_synthetic_pair()
    b[axis][2]+=1e-5
    with pytest.raises(ValueError,match=axis):
        harmonic_comparison(a,b,config)


def test_harmonic_comparison_rejects_a_truncated_film_axis_before_pairwise_errors():
    a,b,config=_synthetic_pair()
    b['film_t_s']=b['film_t_s'][:-1]
    with pytest.raises(ValueError,match='film_t_s'):
        harmonic_comparison(a,b,config)
