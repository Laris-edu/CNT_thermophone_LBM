"""Common observations for a ratio-two solver and its uniform-fine control."""
from dataclasses import astuple
import math
import time

import numpy as np

from core.collision_smrt import collide_fg
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from coupling.planar_open import step_open_film, step_two_level_open_film
from postproc.power import stationarity_at_cycle
from reference.lim2013 import matched_gas
from simulation.study import DEFAULTS


def observe_conserved(state, z_m, probe_z_m, method='linear'):
    """Linear observation of rho, rho*u and total energy at physical positions."""
    z=np.asarray(z_m,dtype=float); probes=np.asarray(probe_z_m,dtype=float)
    if (z.ndim!=1 or probes.ndim!=1 or len(z)<2 or np.any(np.diff(z)<=0)
            or not np.isfinite(probes).all() or np.any(probes<z[0]) or np.any(probes>z[-1])):
        raise ValueError('probes must be finite and inside the physical cell centers')
    if method not in ('linear','nearest'):
        raise ValueError('unknown probe observation method')
    rows=np.argmin(abs(z[:,None]-probes[None,:]),axis=0)
    interp=lambda values: np.interp(probes,z,values) if method=='linear' else values[rows]
    rho=np.asarray(state.rho)[:,0]; u=np.asarray(state.u)[:,0,:]
    observed_rho=interp(rho)
    momentum=np.column_stack([interp(rho*u[:,j]) for j in range(2)])
    observed_u=momentum/observed_rho[:,None]
    energy=interp(np.asarray(state.E_tot)[:,0])
    pressure=(state.gamma-1)*(energy-.5*observed_rho*np.sum(observed_u**2,axis=1))
    return dict(rho=observed_rho,u=observed_u,E_tot=energy,p=pressure,theta=pressure/observed_rho)


def _interface_array(solver):
    audit=getattr(solver,'last_interface_audit',None)
    if audit is None:
        return None
    return np.asarray(audit.as_array() if hasattr(audit,'as_array') else audit,dtype=float)


def collect_synchronized(solver, output, run):
    """Save gas endpoints every two fine steps and every fine wall/film interval.

    A failed partial uniform pair contributes no samples. Interface exchanges
    come from recorded crossing links, never from a storage residual.
    """
    m=solver.mapping; gas=matched_gas(m); dt=m.lattice.dt_s
    mode=run['mode']; multigrid=hasattr(solver,'nf')
    sampling=run['sampling']
    method=sampling.get('probe_interpolation','linear')
    if sampling.get('sync_fine_steps',2)!=2 or method not in ('linear','nearest'):
        raise ValueError('synchronized observations require two fine steps and linear/nearest interpolation')
    z=np.asarray(solver.z_m); widths=np.asarray(getattr(solver,'cell_width_m',np.full(solver.ny,m.lattice.dx_m)))
    levels=np.asarray(getattr(solver,'level_index',np.zeros(solver.ny,dtype=int)))
    H=float(np.sum(widths)); probes=np.asarray(run['probes_m'],dtype=float)
    if method=='nearest':
        probes=z[np.argmin(abs(z[:,None]-probes[None,:]),axis=0)]
    interface=run.get('interface_pulse',run.get('thermal_step',{})).get('interface_m')
    if interface is not None and not multigrid:
        face=int(round(interface/m.lattice.dx_m))
        if not math.isclose(face*m.lattice.dx_m,interface,rel_tol=0,abs_tol=m.lattice.dx_m*1e-8):
            raise ValueError('uniform diagnostic interface must be a cell face')
        solver.interface_face_index=face
    if mode=='interface_pulse':
        pulse=run['interface_pulse']; epsilon=pulse['epsilon']*np.exp(-.5*((z-pulse['center_fraction']*H)/(pulse['width_fraction']*H))**2)
        rho=m.lattice.rho_ref_lu*(1+epsilon[:,None])
        theta=m.theta_ref_lu*(1+epsilon[:,None])**(gas.gamma-1)
        u=np.zeros((solver.ny,1,2)); u[:,0,1]=pulse['direction']*m.c0_lu*epsilon
        solver.initialize_from_macro(rho,u,theta)
        sync_steps=int(pulse['sync_steps'])
    elif mode=='thermal_step':
        sync_steps=int(math.ceil(run['thermal_step']['duration_s']/(2*dt)-1e-12))
    elif mode in ('thermal','power'):
        sync_steps=int(math.ceil(run['cycles']/run['frequency_Hz']/(2*dt)-1e-12))
    else:
        raise ValueError('unsupported synchronized mode')
    state=solver.get_macro()
    observe_conserved(state,z,probes,method)
    initial_energy=float(np.sum(state.E_tot[:,0]*widths)*m.pressure_scale)
    initial_mass=float(np.sum(state.rho[:,0]*widths)/m.lattice.dx_m)
    face=solver.nf if multigrid else getattr(solver,'interface_face_index',None)
    def block_totals(current):
        columns=np.column_stack((current.rho[:,0]*m.rho_scale,
            current.rho[:,0,None]*current.u[:,0,:]*m.rho_scale*m.velocity_scale,
            current.E_tot[:,0]*m.pressure_scale))*widths[:,None]
        return np.array([np.sum(columns[:face],axis=0),np.sum(columns[face:],axis=0)])
    initial_blocks=block_totals(state) if face is not None else None
    block_history=[]; running_audit=np.zeros(6)
    analysis=run.get('analysis',DEFAULTS['analysis'])
    fine_names=('heat_time_s','wall_temperature_K','wall_flux_W_m2','wall_energy_J_m2','wall_mass_lu')
    if mode=='power':
        fine_names+=('film_t_s','film_temperature_K','film_energy_J_m2','power_W_m2','input_energy_J_m2','leak_energy_J_m2')
        film=run['film']; temperature=film['initial_temperature_K']
        params=FilmOdeParams(C_A_si=film['C_A_J_m2K'],T_ref_K=film['T_ref_K'],
            linear_leak_conductance_si=film['G_W_m2K'],gas_flux_factor=film['gas_flux_factor'])
        mean=run['power']['mean_W']/run['power']['area_m2']
        drive=SinusoidalDrive(mean,-mean,run['frequency_Hz'])
        committed_temperature=temperature; running_work=running_leak=0.
    fine={name:[] for name in fine_names}
    gas_names=('t_s','gas_energy_J_m2','mass_lu','first_cell_temperature_K','outlet_temperature_K',
               'outlet_density_kg_m3','outlet_velocity_m_s','outlet_pressure_Pa')
    history={name:[] for name in gas_names}; pressure=[]; velocity=[]; audits=[]; exchanges=[]
    profiles={name:[] for name in ('t_s','temperature_K','pressure_Pa','velocity_m_s','density_kg_m3','conductive_heat_flux_W_m2')}
    probe_heat=[]; min_f=min_g=float('inf'); completed=False; stop_reason='duration_limit'
    def profile(t,current):
        values=dict((name,np.asarray(values).copy()) for name,values in (('temperature_K',current.theta[:,0]*m.temperature_scale),
                ('pressure_Pa',current.p[:,0]*m.pressure_scale),
                ('velocity_m_s',current.u[:,0,1]*m.velocity_scale),
                ('density_kg_m3',current.rho[:,0]*m.rho_scale),
                ('conductive_heat_flux_W_m2',solver.get_heat_flux_lu()[:,0,1]*m.heat_flux_scale)))
        profiles['t_s'].append(t)
        for name,value in values.items():
            profiles[name].append(value)
    profile(0.,state)
    warmup_started=time.perf_counter()
    for mapping in ([m,solver.coarse_mapping] if multigrid else [m]):
        collide_fg(solver.f[:3].copy(),solver.g[:3].copy(),mapping,lattice=solver.lattice,backend=solver.collision_backend)
    warmup_s=time.perf_counter()-warmup_started
    started=time.perf_counter(); advance_s=0.; last_progress=started; last_cycle=0
    profile_spacing=1/(run['frequency_Hz']*sampling.get('profile_samples_per_cycle',32))
    next_profile=profile_spacing
    def wall_temperature(t):
        if mode=='interface_pulse':
            return gas.T0_K
        if mode=='thermal_step':
            return gas.T0_K+run['amplitude_K']
        ramp_cycles=run.get('ramp_cycles',0.)
        ramp=.5*(1-np.cos(np.pi*min(1.,t*run['frequency_Hz']/ramp_cycles))) if ramp_cycles else 1.
        return gas.T0_K+run['amplitude_K']*ramp*np.sin(2*np.pi*run['frequency_Hz']*t)
    def snapshot():
        values={key:np.asarray(value) for key,value in history.items()}
        values.update({key:np.asarray(value) for key,value in fine.items()})
        values.update(pressure_Pa=np.asarray(pressure).reshape(-1,len(probes)),
            velocity_m_s=np.asarray(velocity).reshape(-1,len(probes)))
        return values
    try:
        for k in range(sync_steps):
            pending=[]; pair=[]; pair_exchanges=[]; advance_started=time.perf_counter()
            if multigrid:
                if mode=='power':
                    result=step_two_level_open_film(solver=solver,params=params,T_s_K=temperature,drive=drive)
                    temperature=result.T_s_K; sync_audit=result.audit
                    pending=list(result.steps)
                else:
                    sync_audit=solver.step_temperature(wall_temperature)
                    pending=list(solver.last_substep_audits)
                exchange=_interface_array(solver)
            else:
                for substep in range(2):
                    midpoint=(solver.t_lu+.5)*dt
                    if mode=='power':
                        result=step_open_film(solver=solver,params=params,T_s_K=temperature,drive=drive)
                        temperature=result.T_s_K; audit=result.audit; pending.append(result)
                    else:
                        audit=solver.step_temperature(wall_temperature(midpoint))
                        pending.append((midpoint,audit))
                    if (abs(audit.mass_residual_lu)>=analysis['mass_residual_lu']
                            or abs(audit.energy_residual_J_m2)>=analysis['energy_residual_J_m2']):
                        raise FloatingPointError('fine-step mass/energy audit failed before observation commit')
                    pair.append(np.asarray(astuple(audit),dtype=float))
                    item=_interface_array(solver)
                    if item is not None:
                        pair_exchanges.append(item)
                sync_audit=np.sum(pair,axis=0)
                exchange=np.sum(pair_exchanges,axis=0) if len(pair_exchanges)==2 else None
            advance_s+=time.perf_counter()-advance_started
            state=solver.get_macro(); observed=observe_conserved(state,z,probes,method)
            t=(k+1)*2*dt
            values=(t,float(np.sum(state.E_tot[:,0]*widths)*m.pressure_scale),
                float(np.sum(state.rho[:,0]*widths)/m.lattice.dx_m),
                state.theta[0,0]*m.temperature_scale,state.theta[-1,0]*m.temperature_scale,
                state.rho[-1,0]*m.rho_scale,state.u[-1,0,1]*m.velocity_scale,state.p[-1,0]*m.pressure_scale)
            p=observed['p']*m.pressure_scale; u=observed['u'][:,1]*m.velocity_scale
            af=np.asarray(astuple(sync_audit) if hasattr(sync_audit,'__dataclass_fields__') else sync_audit,dtype=float)
            if not all(np.isfinite(v).all() for v in (values,p,u,af,solver.f,solver.g)) or min(np.min(solver.f),np.min(solver.g))<=0:
                raise FloatingPointError('nonfinite state or nonpositive populations; no clipping applied')
            combined=running_audit+af
            wall_masses=[(item.audit if mode=='power' else item if multigrid else item[1]).wall_mass_lu for item in pending]
            if (max(abs(np.asarray(wall_masses)))>=analysis['wall_mass_lu']
                    or abs(af[4])>=analysis['mass_residual_lu'] or abs(af[5])>=analysis['energy_residual_J_m2']
                    or abs(values[2]-initial_mass-combined[0]-combined[1])>=analysis['cumulative_mass_lu']
                    or abs(values[1]-initial_energy-combined[2]-combined[3])>=analysis['cumulative_energy_J_m2']):
                raise FloatingPointError('synchronized mass/energy audit failed before observation commit')
            if mode=='power':
                previous_temperature=committed_temperature
                film_residuals=[]
                for item in pending:
                    numbers=(item.T_s_K,item.wall_temperature_K,item.heat_time_s,item.power_W_m2,
                        item.input_energy_J_m2,item.leak_energy_J_m2,item.audit.wall_energy_J_m2)
                    if not np.isfinite(numbers).all() or item.T_s_K<=0 or item.wall_temperature_K<=0:
                        raise FloatingPointError('nonfinite or nonpositive fine film/wall observation')
                    film_residuals.append(params.C_A_si*(item.T_s_K-previous_temperature)
                        -item.input_energy_J_m2+item.leak_energy_J_m2
                        +params.gas_flux_factor*item.audit.wall_energy_J_m2)
                    previous_temperature=item.T_s_K
                work=sum(item.input_energy_J_m2 for item in pending)
                leak=sum(item.leak_energy_J_m2 for item in pending)
                factor=params.gas_flux_factor
                previous_gas=history['gas_energy_J_m2'][-1] if history['gas_energy_J_m2'] else initial_energy
                total_step=(params.C_A_si*(previous_temperature-committed_temperature)
                    +factor*(values[1]-previous_gas)-work+leak-factor*af[3])
                film_storage=params.C_A_si*(previous_temperature-film['initial_temperature_K'])
                film_cumulative=film_storage-running_work-work+running_leak+leak+factor*combined[2]
                total_cumulative=(film_storage+factor*(values[1]-initial_energy)
                    -running_work-work+running_leak+leak-factor*combined[3])
                if (max(abs(np.asarray(film_residuals)))>=analysis['energy_residual_J_m2']
                        or abs(total_step)>=analysis['energy_residual_J_m2']
                        or max(abs(film_cumulative),abs(total_cumulative))>=analysis['cumulative_energy_J_m2']):
                    raise FloatingPointError('independent film/total energy audit failed before observation commit')
            # Commit observations only after both fine intervals succeed.
            running_audit=combined
            if mode=='power':
                committed_temperature=previous_temperature
                running_work+=work; running_leak+=leak
            for name,value in zip(gas_names,values):
                history[name].append(value)
            pressure.append(p); velocity.append(u); audits.append(af)
            exchanges.append(np.full((3,4),np.nan) if exchange is None else exchange)
            if face is not None:
                block_history.append(block_totals(state))
            for item in pending:
                if mode=='power':
                    a=item.audit; midpoint=item.heat_time_s; wall_T=item.wall_temperature_K
                elif multigrid:
                    a=item; midpoint=item.heat_time_s; wall_T=wall_temperature(midpoint)
                else:
                    midpoint,a=item; wall_T=wall_temperature(midpoint)
                fine_values=[midpoint,wall_T,a.wall_energy_J_m2/dt,a.wall_energy_J_m2,a.wall_mass_lu]
                if mode=='power':
                    fine_values.extend((midpoint+.5*dt,item.T_s_K,params.C_A_si*(item.T_s_K-params.T_ref_K),
                        item.power_W_m2,item.input_energy_J_m2,item.leak_energy_J_m2))
                for name,value in zip(fine_names,fine_values):
                    fine[name].append(value)
            min_f=min(min_f,float(np.min(solver.f))); min_g=min(min_g,float(np.min(solver.g)))
            if mode=='thermal_step':
                probe_heat.append(np.interp(probes,z,solver.get_heat_flux_lu()[:,0,1])*m.heat_flux_scale)
            if t>=next_profile-1e-12*dt:
                profile(t,state); next_profile=(math.floor(t/profile_spacing)+1)*profile_spacing
            cycle=int(math.floor(t*run['frequency_Hz']+1e-10))
            if mode=='power' and cycle>last_cycle:
                last_cycle=cycle; stop=run['stopping']
                if stop['stop_when_stable'] and cycle>=stop['first_stable_cycle']:
                    diagnostic=stationarity_at_cycle(snapshot(),run['frequency_Hz'],cycle,run['analysis'])
                    if diagnostic['passed']:
                        stop_reason='periodic_stable'; break
            if time.perf_counter()-last_progress>=45:
                last_progress=time.perf_counter()
                print(f'  sync {k+1}/{sync_steps}, t={t:g} s, elapsed={last_progress-started:.1f} s',flush=True)
        if history['t_s'] and profiles['t_s'][-1]!=history['t_s'][-1]:
            profile(history['t_s'][-1],state)
        completed=True
    finally:
        counts=[len(audits),len(pressure),len(velocity),len(exchanges),
            *(len(values) for values in history.values()),*(len(values)//2 for values in fine.values())]
        if face is not None:
            counts.append(len(block_history))
        if mode=='thermal_step':
            counts.append(len(probe_heat))
        count=min(counts)
        for values in history.values():
            del values[count:]
        for values in fine.values():
            del values[2*count:]
        for values in (audits,pressure,velocity,exchanges,block_history,probe_heat):
            del values[count:]
        profile_count=min(len(values) for values in profiles.values())
        last_time=history['t_s'][-1] if count else 0.
        profile_count=min(profile_count,sum(t<=last_time for t in profiles['t_s']))
        for values in profiles.values():
            del values[profile_count:]
        data=snapshot()
        data.update(audit=np.asarray(audits).reshape(-1,6),
            interface_exchange=np.asarray(exchanges).reshape(-1,3,4),
            interface_exchange_available=bool(exchanges and np.isfinite(exchanges).all()),
            initial_mass_lu=initial_mass,initial_gas_energy_J_m2=initial_energy,
            initial_mass_kg_m2=initial_mass*m.rho_scale*m.lattice.dx_m,
            sync_fine_step=2*np.arange(1,count+1,dtype=int),probe_z_m=probes,z_m=z,
            cell_width_m=widths,level_index=levels,actual_length_m=H,
            min_f=min_f,min_g=min_g,run_completed=completed,
            stop_reason=stop_reason if completed else 'failed_or_interrupted',
            warmup_s=warmup_s,advance_s=advance_s,elapsed_s=time.perf_counter()-started,
            level_steps=np.array([solver.level_steps['fine'],solver.level_steps['coarse']] if multigrid else [solver.t_lu,0],dtype=int),
            cell_updates=int(solver.cell_updates if multigrid else solver.t_lu*solver.ny),
            work_count_scope='successful level steps; attempted rolled-back collisions excluded')
        data['mass_kg_m2']=data['mass_lu']*m.rho_scale*m.lattice.dx_m
        if face is not None:
            data.update(block_conserved=np.asarray(block_history).reshape(-1,2,4),initial_block_conserved=initial_blocks)
        if mode=='interface_pulse':
            dp=data['pressure_Pa']-gas.p0_Pa
            rhocu=gas.rho0_kg_m3*gas.c0_m_s*data['velocity_m_s']
            data.update(outgoing_Pa=.5*(dp+rhocu),incoming_Pa=.5*(dp-rhocu))
        for name,values in profiles.items():
            data['profile_'+name]=np.asarray(values) if name=='t_s' else np.asarray(values).reshape(-1,solver.ny)
        if mode=='thermal_step':
            data['conductive_heat_flux_W_m2']=np.asarray(probe_heat).reshape(-1,len(probes))
        if interface is not None:
            data['interface_m']=float(interface)
        if mode=='power':
            data.update(initial_film_temperature_K=film['initial_temperature_K'],
                initial_film_energy_J_m2=params.C_A_si*(film['initial_temperature_K']-params.T_ref_K))
        np.savez_compressed(output/'trajectory.npz',**data)
    return data
