"""Startup diagnostics for actual-exchange, power-driven open-plane data."""
import numpy as np

from postproc.diagnostics import compare, record
from postproc.multiharmonic_fit import fit_multiharmonic
from reference.lim2013 import matched_gas, film_temperature_hat, exact_pressure_hat


def _channels(data):
    yield 'film_temperature', data.get('film_t_s',data['t_s']), data['film_temperature_K']
    yield 'wall_flux', data['heat_time_s'], data['wall_flux_W_m2']
    for j in range(data['pressure_Pa'].shape[1]):
        yield f'pressure_{j}', data['t_s'], data['pressure_Pa'][:,j]
        yield f'velocity_{j}', data['t_s'], data['velocity_m_s'][:,j]


def _fit(t,x,frequency,start,end,order=1):
    t,x=np.asarray(t),np.asarray(x)
    mask=(t*frequency>=start-1e-10)&(t*frequency<end-1e-10)
    values=x[mask]
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite signal in fit window')
    fit=fit_multiharmonic(t[mask],values,2*np.pi*frequency,n_harmonics=3,detrend_order=order)
    z=fit.harmonic(1)
    resolution=256*np.finfo(float).eps*max(1.,float(np.max(abs(values))))
    if not np.isfinite(z) or abs(z)<=resolution or fit.condition_number>1e8:
        raise ValueError('zero/unresolved harmonic or degenerate fit')
    return fit,dict(**record(z),mean=float(np.mean(values)),
        trend_at_midpoint=float(fit.trend_coeffs_scaled[0]),
        drift_per_s=float(fit.trend_coeffs_raw[1]),
        phase_relative_power_deg=float(np.angle(-z,deg=True)),
        residual_rms=float(fit.residual_rms),condition_number=float(fit.condition_number),
        harmonics=[record(fit.harmonic(n)) for n in (1,2,3)])


def stationarity_at_cycle(data, frequency, cycle, analysis):
    """Four disjoint two-cycle windows plus duration/detrend sensitivity.

    This is a finite observation-window criterion, not thermal equilibrium.
    All diagnostics are recomputable from raw endpoint/midpoint samples.
    """
    eligible=cycle>=8
    result=dict(cycle=int(cycle),time_s=float(cycle/frequency),eligible=eligible,
                observation_cycles=[max(0,cycle-8),cycle],channels={},passed=False)
    for name,t,x in _channels(data):
        try:
            latest, item=_fit(t,x,frequency,cycle-2,cycle)
            item['passed']=False
            if eligible:
                fits=[_fit(t,x,frequency,cycle-8+2*j,cycle-6+2*j)[0] for j in range(4)]
                hats=np.array([f.harmonic(1) for f in fits])
                amp_span=float(np.ptp(abs(hats))/abs(hats[-1]))
                phase_span=float(np.degrees(np.ptp(np.unwrap(np.angle(hats)))))
                long,_=_fit(t,x,frequency,cycle-4,cycle)
                curved,_=_fit(t,x,frequency,cycle-2,cycle,order=2)
                sensitivities=[compare(f.harmonic(1),latest.harmonic(1)) for f in (long,curved)]
                item.update(amplitude_span_relative=amp_span,phase_span_deg=phase_span,
                    duration_sensitivity=sensitivities[0],detrend_sensitivity=sensitivities[1],
                    passed=bool(amp_span<analysis['stationary_relative']
                        and phase_span<analysis['stationary_phase_deg']
                        and all(abs(s['amplitude_error_percent'])<100*analysis['stationary_relative']
                            and abs(s['phase_error_deg'])<analysis['stationary_phase_deg'] for s in sensitivities)))
            result['channels'][name]=item
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            result['channels'][name]=dict(passed=False,error=str(exc))
    result['passed']=bool(eligible and all(c['passed'] for c in result['channels'].values()))
    return result


def energy_balances(data, film):
    """Independent storage differences; top exchange is positive into gas."""
    audit=np.asarray(data['audit'])
    film_storage=film['C_A_J_m2K']*(data['film_temperature_K']-float(data['initial_film_temperature_K']))
    gas_storage=data['gas_energy_J_m2']-float(data['initial_gas_energy_J_m2'])
    work=data['input_energy_J_m2']; leak=data['leak_energy_J_m2']
    wall=audit[:,2]; outlet=audit[:,3]; factor=film['gas_flux_factor']
    if 'sync_fine_step' in data:
        ends=np.asarray(data['sync_fine_step'],dtype=int)
        if (len(ends)!=len(audit) or len(ends)==0 or np.any(np.diff(ends)<=0)
                or ends[-1]!=len(film_storage)):
            raise ValueError('sync_fine_step must partition every fine film interval')
        fine_wall=np.asarray(data['wall_energy_J_m2'])
        film_step=np.diff(film_storage,prepend=0.)-work+leak+factor*fine_wall
        film_cumulative=film_storage-np.cumsum(work-leak-factor*fine_wall)
        storage_at_sync=film_storage[ends-1]
        def aggregate(values):
            return np.diff(np.concatenate(([0.],np.cumsum(values)[ends-1])))
        df=np.diff(storage_at_sync,prepend=0.); dg=np.diff(gas_storage,prepend=0.)
        sync_work=aggregate(work); sync_leak=aggregate(leak)
        return dict(film_storage=film_storage,gas_storage=gas_storage,
            film_step=film_step,gas_step=dg-wall-outlet,
            total_step=df+factor*dg-sync_work+sync_leak-factor*outlet,
            film_cumulative=film_cumulative,
            gas_cumulative=gas_storage-np.cumsum(wall+outlet),
            total_cumulative=storage_at_sync+factor*gas_storage-np.cumsum(sync_work-sync_leak+factor*outlet))
    df=np.diff(film_storage,prepend=0.); dg=np.diff(gas_storage,prepend=0.)
    return dict(film_storage=film_storage,gas_storage=gas_storage,
        film_step=df-work+leak+factor*wall,gas_step=dg-wall-outlet,
        total_step=df+factor*dg-work+leak-factor*outlet,
        film_cumulative=film_storage-np.cumsum(work-leak-factor*wall),
        gas_cumulative=gas_storage-np.cumsum(wall+outlet),
        total_cumulative=film_storage+factor*gas_storage-np.cumsum(work-leak+factor*outlet))


def analyze_power(data,mapping,run,previous=None):
    frequency=run['frequency_Hz']; gas=matched_gas(mapping); a=run['analysis']
    cycles=float(data['t_s'][-1]*frequency)
    history=[stationarity_at_cycle(data,frequency,n,a)
             for n in range(run['stopping']['first_diagnostic_cycle'],int(np.floor(cycles+1e-10))+1)]
    eligible=[h for h in history if h['cycle']>=run['stopping']['first_stable_cycle']]
    established=next((h for h in eligible if h['passed']),None)
    last=history[-1] if history else None
    stationary=bool(eligible and eligible[-1]['passed'])
    balances=energy_balances(data,run['film']); audit=data['audit']
    maximum=lambda values: float(np.max(abs(values)))
    result=dict(kind='power_driven_open_planar_local_source',frequency_Hz=frequency,
        air=dict(T0_K=gas.T0_K,rho0_kg_m3=gas.rho0_kg_m3,c0_m_s=gas.c0_m_s,
            kappa_W_mK=gas.kappa_W_mK,cp_J_kgK=gas.cp_J_kgK,
            alpha_m2_s=gas.alpha_m2_s,p0_Pa=gas.p0_Pa),
        electrical_frequency_Hz=frequency/2,mean_power_W=run['power']['mean_W'],
        film_area_m2=run['power']['area_m2'],film=run['film'],cycles=cycles,
        requested_max_cycles=run['cycles'],stop_reason=str(data['stop_reason']),
        elapsed_s=float(data['elapsed_s']),stationarity=history,
        establishment_time_s=established['time_s'] if established else None,
        establishment_observation_cycles=established['observation_cycles'] if established else None,
        periodic_response_stable=stationary,thermal_equilibrium_assessed=False,
        validation_scope='energy/state/finite-window stationarity at this grid and domain only',
        grid_independence_assessed=False,domain_independence_assessed=False,
        max_wall_mass_transfer_lu=maximum(audit[:,0]),
        max_mass_residual_lu=maximum(audit[:,4]),
        max_energy_residual_J_m2=maximum(balances['gas_step']),
        max_film_energy_residual_J_m2=maximum(balances['film_step']),
        max_total_energy_residual_J_m2=maximum(balances['total_step']),
        cumulative_mass_residual_lu=maximum(data['mass_lu']-float(data['initial_mass_lu'])-np.cumsum(audit[:,0]+audit[:,1])),
        cumulative_energy_residual_J_m2=maximum(balances['gas_cumulative']),
        cumulative_film_energy_residual_J_m2=maximum(balances['film_cumulative']),
        cumulative_total_energy_residual_J_m2=maximum(balances['total_cumulative']),
        min_f=float(data['min_f']),min_g=float(data['min_g']),
        final_film_temperature_K=float(data['film_temperature_K'][-1]),
        max_film_temperature_K=float(np.max(data['film_temperature_K'])),
        probe_z_m=np.asarray(data['probe_z_m']).tolist(),
        outlet=dict(max_abs_temperature_change_K=maximum(data['outlet_temperature_K']-gas.T0_K),
            max_abs_density_change_kg_m3=maximum(data['outlet_density_kg_m3']-gas.rho0_kg_m3),
            max_abs_velocity_m_s=maximum(data['outlet_velocity_m_s']),
            max_abs_pressure_change_Pa=maximum(data['outlet_pressure_Pa']-gas.p0_Pa)),
        energy_final_J_m2=dict(film_storage=float(balances['film_storage'][-1]),
            one_side_gas_storage=float(balances['gas_storage'][-1]),
            input=float(np.sum(data['input_energy_J_m2'])),leak=float(np.sum(data['leak_energy_J_m2'])),
            one_side_wall=float(np.sum(audit[:,2])),one_side_outlet_into_gas=float(np.sum(audit[:,3]))))
    result['gates']=dict(periodic_response=stationary,
        impermeable_wall=result['max_wall_mass_transfer_lu']<a['wall_mass_lu'],
        mass_audit=result['max_mass_residual_lu']<a['mass_residual_lu'],
        cumulative_mass_audit=result['cumulative_mass_residual_lu']<a['cumulative_mass_lu'],
        energy_audit=max(result[k] for k in ('max_energy_residual_J_m2','max_film_energy_residual_J_m2','max_total_energy_residual_J_m2'))<a['energy_residual_J_m2'],
        cumulative_energy_audit=max(result[k] for k in ('cumulative_energy_residual_J_m2','cumulative_film_energy_residual_J_m2','cumulative_total_energy_residual_J_m2'))<a['cumulative_energy_J_m2'],
        positive_populations=min(result['min_f'],result['min_g'])>0,
        finite_state=bool(all(np.isfinite(data[key]).all() for key in
            ('film_temperature_K','pressure_Pa','velocity_m_s','gas_energy_J_m2','audit'))))
    # Independently driven reference; the 300 K background is diagnostic only.
    film=run['film']; phat=-run['power']['mean_W']/run['power']['area_m2']
    tref=film_temperature_hat(frequency,phat,film['C_A_J_m2K'],
                             beta0_per_side_W_m2K=film['G_W_m2K']/2,gas=gas)
    refs={'film_temperature':tref}
    refs.update({f'pressure_{j}':exact_pressure_hat(frequency,tref,float(z),gas=gas)
                 for j,z in enumerate(data['probe_z_m'])})
    result['linear_reference_diagnostic']=dict(background_temperature_K=gas.T0_K,
        is_acceptance_gate=False,channels={})
    for name,reference in refs.items():
        item=dict(reference=record(reference))
        if last and 're' in last['channels'][name]:
            numerical=last['channels'][name]
            item['comparison']=compare(complex(numerical['re'],numerical['im']),reference)
        result['linear_reference_diagnostic']['channels'][name]=item
    return result
