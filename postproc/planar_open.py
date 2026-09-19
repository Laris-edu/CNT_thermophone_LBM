"""Open-plane diagnostics from saved arrays; never advances the gas solver."""
import numpy as np

from postproc.diagnostics import compare, fit_window, record
from reference.lim2013 import conductive_heat_flux_hat, exact_pressure_hat, matched_gas


def analyze_thermal(data, mapping, run, previous=None):
    previous=previous or {}
    gas=matched_gas(mapping)
    frequency=run['frequency_Hz']; amplitude=run['amplitude_K']; cycles=run['cycles']
    analysis=run['analysis']; window=analysis['window_cycles']
    def fit(t,x,start,end):
        return fit_window(t,x,frequency,start,end,harmonics=analysis['harmonics'],
                          detrend_order=analysis['detrend_order'])
    q_hat,q_quality=fit(data['heat_time_s'],data['wall_flux_W_m2'],cycles-window,cycles)
    q_previous,_=fit(data['heat_time_s'],data['wall_flux_W_m2'],cycles-2*window,cycles-window)
    q_ref=conductive_heat_flux_hat(frequency,-1j*amplitude,gas=gas)
    probes=[]
    for j,z in enumerate(data['probe_z_m']):
        p_hat,quality=fit(data['t_s'],data['pressure_Pa'][:,j],cycles-window,cycles)
        old,_=fit(data['t_s'],data['pressure_Pa'][:,j],cycles-2*window,cycles-window)
        p_ref=exact_pressure_hat(frequency,-1j*amplitude,z,gas=gas)
        probes.append(dict(z_m=float(z),lbm=record(p_hat),lim=record(p_ref),
            comparison=compare(p_hat,p_ref),fit=quality,cycle_change=float(abs(p_hat/old-1))))
    a=data['audit']
    min_f=float(data['min_f']) if 'min_f' in data else previous['min_f']
    min_g=float(data['min_g']) if 'min_g' in data else previous['min_g']
    result=dict(kind='prescribed_temperature_direct_pressure',frequency_Hz=frequency,
        wall_amplitude_K=amplitude,cycles=cycles,ramp_cycles=run['ramp_cycles'],probes=probes,
        heat_flux=dict(lbm=record(q_hat),lim=record(q_ref),comparison=compare(q_hat,q_ref),fit=q_quality,
            cycle_change=float(abs(q_hat/q_previous-1))),
        max_wall_mass_transfer_lu=float(np.max(abs(a[:,0]))),
        max_mass_residual_lu=float(np.max(abs(a[:,4]))),
        max_energy_residual_J_m2=float(np.max(abs(a[:,5]))),
        cumulative_mass_residual_lu=float(np.max(abs(data['mass_lu']-data['initial_mass_lu']-np.cumsum(a[:,0]+a[:,1])))),
        cumulative_energy_residual_J_m2=float(np.max(abs(data['gas_energy_J_m2']-data['initial_gas_energy_J_m2']-np.cumsum(a[:,2]+a[:,3])))),
        min_f=min_f,min_g=min_g,
        elapsed_s=float(data['elapsed_s']) if 'elapsed_s' in data else previous.get('elapsed_s',0.))
    result['gates']=dict(
        pressure_amplitude=all(abs(p['comparison']['amplitude_error_percent'])<analysis['amplitude_percent'] for p in probes),
        pressure_phase=all(abs(p['comparison']['phase_error_deg'])<analysis['phase_deg'] for p in probes),
        cycle_stationarity=max([p['cycle_change'] for p in probes]+[result['heat_flux']['cycle_change']])<analysis['stationary_relative'],
        impermeable_wall=result['max_wall_mass_transfer_lu']<analysis['wall_mass_lu'],
        mass_audit=result['max_mass_residual_lu']<analysis['mass_residual_lu'],
        energy_audit=result['max_energy_residual_J_m2']<analysis['energy_residual_J_m2'],
        cumulative_mass_audit=result['cumulative_mass_residual_lu']<analysis['cumulative_mass_lu'],
        cumulative_energy_audit=result['cumulative_energy_residual_J_m2']<analysis['cumulative_energy_J_m2'],
        positive_populations=min(min_f,min_g)>0)
    return result


def analyze_pulse(data, mapping, run, previous=None):
    previous=previous or {}
    gas=matched_gas(mapping); pulse=run['pulse']
    H=float(data['actual_length_m']) if 'actual_length_m' in data else previous['actual_length_m']
    z=float(data['probe_z_m']) if 'probe_z_m' in data else previous['probe_m']
    ti=(z-pulse['center_fraction']*H)/gas.c0_m_s
    tr=(2*H-z-pulse['center_fraction']*H)/gas.c0_m_s
    halfwindow=pulse['window_fraction']*H/gas.c0_m_s
    incident=float(np.max(abs(data['outgoing_Pa'][abs(data['t_s']-ti)<halfwindow])))
    reflected=float(np.max(abs(data['incoming_Pa'][abs(data['t_s']-tr)<halfwindow])))
    return dict(kind='outgoing_gaussian_pulse',probe_m=z,incident_Pa=incident,reflected_Pa=reflected,
        reflection_ratio=reflected/incident,pass_reflection=bool(reflected/incident<run['analysis']['reflection_ratio']),
        max_mass_residual_lu=float(np.max(abs(data['audit'][:,4]))) if 'audit' in data else previous['max_mass_residual_lu'],
        max_energy_residual_J_m2=float(np.max(abs(data['audit'][:,5]))) if 'audit' in data else previous['max_energy_residual_J_m2'])
