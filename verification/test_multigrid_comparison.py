"""The research comparison measures untranslated, time-domain waveforms."""
from copy import deepcopy
import numpy as np

from Compute.multi_grid.compare import pulse_comparison
from core.unit_mapping import create_unit_mapping
from reference.lim2013 import matched_gas
from simulation.study import ROOT, load_study


def test_extra_reverse_wave_and_time_delay_cannot_be_hidden_by_peak_matching():
    config=load_study(ROOT/'Compute/multi_grid/study.yaml')[0]
    gas=matched_gas(create_unit_mapping(config)); run=config['run']; p=run['interface_pulse']
    H=run['length_m']; source=p['center_fraction']*H; z=np.array(run['probes_m'])
    t=np.arange(1,389)*2*config['lattice']['dt_s']; width=p['width_fraction']*H/gas.c0_m_s
    wave=np.exp(-.5*((t[:,None]-(z-source)/gas.c0_m_s)/width)**2)
    data=dict(t_s=t,probe_z_m=z,actual_length_m=H,
        pressure_Pa=gas.p0_Pa+wave,velocity_m_s=wave/(gas.rho0_kg_m3*gas.c0_m_s))
    assert all(pulse_comparison(data,data,config)['gates'].values())
    reflected=.02*np.exp(-.5*((t-(2*p['interface_m']-source-z[0])/gas.c0_m_s)/width)**2)
    bad=deepcopy(data)
    bad['pressure_Pa'][:,0]+=reflected
    bad['velocity_m_s'][:,0]-=reflected/(gas.rho0_kg_m3*gas.c0_m_s)
    result=pulse_comparison(bad,data,config)
    assert not result['gates']['reflection'] and not result['gates']['additional_reverse']
    delayed=deepcopy(data)
    delayed['pressure_Pa'][:,1]=gas.p0_Pa+np.roll(wave[:,1],2)
    delayed['velocity_m_s'][:,1]=np.roll(wave[:,1],2)/(gas.rho0_kg_m3*gas.c0_m_s)
    result=pulse_comparison(delayed,data,config)
    assert result['gates']['transmission_peak']
    assert not result['gates']['transmission_unshifted_l2']
