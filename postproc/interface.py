"""Time-domain interface diagnostics from saved physical observations."""
import numpy as np

from reference.lim2013 import matched_gas


def _physical(data,mapping,analysis):
    audit=np.asarray(data['audit']); maximum=lambda x: float(np.max(np.abs(x)))
    mass=np.asarray(data['mass_lu'])-float(data['initial_mass_lu'])-np.cumsum(audit[:,0]+audit[:,1])
    energy=np.asarray(data['gas_energy_J_m2'])-float(data['initial_gas_energy_J_m2'])-np.cumsum(audit[:,2]+audit[:,3])
    stats=dict(max_wall_mass_transfer_lu=maximum(audit[:,0]),max_mass_residual_lu=maximum(audit[:,4]),
        max_energy_residual_J_m2=maximum(audit[:,5]),cumulative_mass_residual_lu=maximum(mass),
        cumulative_energy_residual_J_m2=maximum(energy),min_f=float(data['min_f']),min_g=float(data['min_g']))
    gates=dict(impermeable_wall=stats['max_wall_mass_transfer_lu']<analysis['wall_mass_lu'],
        mass_audit=stats['max_mass_residual_lu']<analysis['mass_residual_lu'],
        energy_audit=stats['max_energy_residual_J_m2']<analysis['energy_residual_J_m2'],
        cumulative_mass_audit=stats['cumulative_mass_residual_lu']<analysis['cumulative_mass_lu'],
        cumulative_energy_audit=stats['cumulative_energy_residual_J_m2']<analysis['cumulative_energy_J_m2'],
        positive_populations=min(stats['min_f'],stats['min_g'])>0,
        finite_state=all(np.isfinite(data[key]).all() for key in ('pressure_Pa','velocity_m_s','gas_energy_J_m2','audit')))
    available=bool(data.get('interface_exchange_available',False))
    stats['interface_exchange_available']=available
    if available:
        exchange=np.asarray(data['interface_exchange']); residual=exchange[:,0]+exchange[:,1]
        scale=mapping.rho_scale*mapping.lattice.dx_m
        stats['max_interface_residual_SI']=np.max(abs(residual),axis=0).tolist()
        gates['interface_mass_audit']=maximum(residual[:,0])/scale<analysis['mass_residual_lu']
        gates['interface_energy_audit']=maximum(residual[:,3])<analysis['energy_residual_J_m2']
        # Momentum is reported in SI; no undeclared momentum tolerance is invented.
        if 'block_conserved' in data:
            stored=np.asarray(data['block_conserved'])-np.asarray(data['initial_block_conserved'])
            boundary=np.zeros_like(stored)
            boundary[:,0,0]=audit[:,0]*scale; boundary[:,1,0]=audit[:,1]*scale
            boundary[:,0,3]=audit[:,2]; boundary[:,1,3]=audit[:,3]
            residual_blocks=stored-np.cumsum(boundary+exchange[:,:2,:],axis=0)
            stats['cumulative_block_mass_residual_kg_m2']=maximum(residual_blocks[:,:,0])
            stats['cumulative_block_energy_residual_J_m2']=maximum(residual_blocks[:,:,3])
            gates['block_mass_audit']=maximum(residual_blocks[:,:,0])/scale<analysis['cumulative_mass_lu']
            gates['block_energy_audit']=maximum(residual_blocks[:,:,3])<analysis['cumulative_energy_J_m2']
    return stats,gates


def analyze_interface_pulse(data,mapping,run,previous=None):
    gas=matched_gas(mapping); pulse=run['interface_pulse']; direction=pulse['direction']
    H=float(data['actual_length_m']); xi=pulse['interface_m']; z=np.asarray(data['probe_z_m'])
    left=np.flatnonzero(z<xi); right=np.flatnonzero(z>xi)
    if direction not in (-1,1) or not len(left) or not len(right):
        raise ValueError('interface pulse requires a direction and probes on both sides')
    near_left=left[np.argmax(z[left])]; near_right=right[np.argmin(z[right])]
    source=pulse['center_fraction']*H; upstream=near_left if direction==1 else near_right
    downstream=near_right if direction==1 else near_left
    ti=direction*(z[upstream]-source)/gas.c0_m_s
    tr=direction*(2*xi-source-z[upstream])/gas.c0_m_s
    tt=direction*(z[downstream]-source)/gas.c0_m_s
    half=4*pulse['width_fraction']*H/gas.c0_m_s
    t=np.asarray(data['t_s']); dp=np.asarray(data['pressure_Pa'])-gas.p0_Pa
    rhocu=gas.rho0_kg_m3*gas.c0_m_s*np.asarray(data['velocity_m_s'])
    forward=.5*(dp+direction*rhocu); backward=.5*(dp-direction*rhocu)
    def peak(values,center):
        window=(t>=center-half)&(t<=center+half)
        if center-half<0 or np.count_nonzero(window)<3 or t[-1]<center+half:
            raise ValueError('interface pulse analysis windows must be fully sampled')
        indices=np.flatnonzero(window); index=indices[np.argmax(abs(values[window]))]
        return float(abs(values[index])),float(t[index]),int(len(indices))
    incident,incident_peak,ni=peak(forward[:,upstream],ti)
    reflected,reflected_peak,nr=peak(backward[:,upstream],tr)
    transmitted,transmitted_peak,nt=peak(forward[:,downstream],tt)
    if incident<=0:
        raise ValueError('interface incident pulse is unresolved')
    ratio=reflected/incident; passed=bool(ratio<run['analysis']['reflection_ratio'])
    stats,physical=_physical(data,mapping,run['analysis'])
    return dict(kind='interface_gaussian_pulse',direction=direction,interface_m=xi,
        incident_probe_m=float(z[upstream]),transmitted_probe_m=float(z[downstream]),
        incident_Pa=incident,reflected_Pa=reflected,transmitted_Pa=transmitted,
        incident_time_s=ti,reflected_time_s=tr,transmitted_time_s=tt,
        incident_peak_time_s=incident_peak,reflected_peak_time_s=reflected_peak,transmitted_peak_time_s=transmitted_peak,
        window_halfwidth_s=half,window_sample_counts=[ni,nr,nt],reflection_ratio=ratio,
        pass_reflection=passed,gates=dict(reflection=passed),physical_gates=physical,
        physical_validation_passed=bool(all(physical.values())),**stats)


def analyze_thermal_step(data,mapping,run,previous=None):
    stats,gates=_physical(data,mapping,run['analysis'])
    available=bool(data.get('interface_exchange_available',False))
    gates['interface_exchange_recorded']=available
    result=dict(kind='prescribed_temperature_step',interface_m=run['thermal_step']['interface_m'],
        requested_duration_s=run['thermal_step']['duration_s'],actual_duration_s=float(data['t_s'][-1]),
        wall_amplitude_K=run['amplitude_K'],gates=gates,**stats)
    if available:
        result['integrated_interface_exchange_SI']=np.sum(data['interface_exchange'],axis=0).tolist()
    result['final_temperature_K']=np.asarray(data['profile_temperature_K'])[-1].tolist()
    result['final_conductive_heat_flux_W_m2']=np.asarray(data['profile_conductive_heat_flux_W_m2'])[-1].tolist()
    result['elapsed_s']=float(data['elapsed_s'])
    return result
