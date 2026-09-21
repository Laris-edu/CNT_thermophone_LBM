"""Cold-start power run, using the public conservative open-film step."""
from dataclasses import astuple
import time

import numpy as np

from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from coupling.planar_open import step_open_film
from postproc.power import stationarity_at_cycle


def collect_power(solver, output, run):
    mapping=solver.mapping; dt=mapping.lattice.dt_s; frequency=run['frequency_Hz']
    film=run['film']
    params=FilmOdeParams(C_A_si=film['C_A_J_m2K'],T_ref_K=film['T_ref_K'],
        linear_leak_conductance_si=film['G_W_m2K'],gas_flux_factor=film['gas_flux_factor'])
    mean=run['power']['mean_W']/run['power']['area_m2']
    drive=SinusoidalDrive(mean,-mean,frequency)
    rows=np.array([np.argmin(abs(solver.z_m-z)) for z in run['probes_m']])
    steps=int(np.ceil(run['cycles']/frequency/dt)); count=len(rows)
    fields=('t_s','heat_time_s','film_temperature_K','wall_temperature_K','power_W_m2',
        'input_energy_J_m2','leak_energy_J_m2','wall_flux_W_m2','gas_energy_J_m2',
        'mass_lu','film_energy_J_m2','first_cell_temperature_K','outlet_temperature_K',
        'outlet_density_kg_m3','outlet_velocity_m_s','outlet_pressure_Pa')
    history=np.empty((steps,len(fields)+2*count)); audits=np.empty((steps,6))
    state=solver.get_macro()
    initial_energy=float(np.sum(state.E_tot)*mapping.pressure_scale*mapping.lattice.dx_m)
    initial_mass=float(np.sum(state.rho)); initial_temperature=film['initial_temperature_K']
    temperature=initial_temperature
    n=0; min_f=min_g=float('inf'); completed=False; stop_reason='cycle_limit'
    started=time.perf_counter(); last_progress=started; last_cycle=0; next_profile=1
    profile_times=[]; profiles={name:[] for name in ('temperature_K','pressure_Pa','density_kg_m3','velocity_m_s')}

    def save_profile(t,state):
        profile_times.append(t)
        profiles['temperature_K'].append((state.theta[:,0]*mapping.temperature_scale).copy())
        profiles['pressure_Pa'].append((state.p[:,0]*mapping.pressure_scale).copy())
        profiles['density_kg_m3'].append((state.rho[:,0]*mapping.rho_scale).copy())
        profiles['velocity_m_s'].append((state.u[:,0,1]*mapping.velocity_scale).copy())

    def samples():
        values={name:history[:n,j] for j,name in enumerate(fields)}
        values.update(pressure_Pa=history[:n,len(fields):len(fields)+count],
                      velocity_m_s=history[:n,len(fields)+count:])
        return values

    save_profile(0.,state)
    try:
        for i in range(steps):
            step=step_open_film(solver=solver,params=params,T_s_K=temperature,drive=drive)
            temperature=step.T_s_K; state=solver.get_macro(); t=(i+1)*dt
            min_f=min(min_f,float(np.min(solver.f))); min_g=min(min_g,float(np.min(solver.g)))
            history[i]=[t,step.heat_time_s,temperature,step.wall_temperature_K,step.power_W_m2,
                step.input_energy_J_m2,step.leak_energy_J_m2,step.audit.wall_energy_J_m2/dt,
                float(np.sum(state.E_tot)*mapping.pressure_scale*mapping.lattice.dx_m),
                float(np.sum(state.rho)),params.C_A_si*(temperature-params.T_ref_K),
                state.theta[0,0]*mapping.temperature_scale,state.theta[-1,0]*mapping.temperature_scale,
                state.rho[-1,0]*mapping.rho_scale,state.u[-1,0,1]*mapping.velocity_scale,
                state.p[-1,0]*mapping.pressure_scale,
                *(state.p[rows,0]*mapping.pressure_scale),*(state.u[rows,0,1]*mapping.velocity_scale)]
            audits[i]=astuple(step.audit); n=i+1
            if not np.all(np.isfinite(history[i])) or min(min_f,min_g)<=0:
                raise FloatingPointError('nonfinite state or nonpositive populations; no clipping applied')
            if (i+1)%1024==0 and time.perf_counter()-last_progress>=45:
                last_progress=time.perf_counter()
                print(f'  step {i+1}/{steps}, cycle={t*frequency:.3f}, '
                      f'Tfilm={temperature:.6f} K',flush=True)
            profile_index=t*frequency*run['sampling']['profile_samples_per_cycle']
            if profile_index>=next_profile-1e-10:
                save_profile(t,state)
                next_profile=int(np.floor(profile_index+1e-10))+1
            cycle=int(np.floor(t*frequency+1e-10))
            if cycle>last_cycle:
                last_cycle=cycle
                diagnostic=None
                if cycle>=run['stopping']['first_diagnostic_cycle']:
                    diagnostic=stationarity_at_cycle(samples(),frequency,cycle,run['analysis'])
                stable=bool(diagnostic and diagnostic['passed']
                            and cycle>=run['stopping']['first_stable_cycle'])
                print(f'  cycle {cycle}/{run["cycles"]:g}, Tfilm={temperature:.6f} K, '
                      f'periodic_stable={stable}, elapsed={time.perf_counter()-started:.1f} s',flush=True)
                if run['stopping']['stop_when_stable'] and stable:
                    stop_reason='periodic_stable'
                    break
        completed=True
    finally:
        data=samples()
        data.update(audit=audits[:n],initial_mass_lu=initial_mass,
            initial_gas_energy_J_m2=initial_energy,initial_film_temperature_K=initial_temperature,
            initial_film_energy_J_m2=params.C_A_si*(initial_temperature-params.T_ref_K),
            probe_z_m=solver.z_m[rows],z_m=solver.z_m,
            actual_length_m=solver.ny*mapping.lattice.dx_m,
            min_f=min_f,min_g=min_g,run_completed=completed,
            stop_reason=stop_reason if completed else 'failed_or_interrupted',
            elapsed_s=time.perf_counter()-started,profile_t_s=np.asarray(profile_times))
        data.update({f'profile_{name}':np.asarray(values).reshape(-1,solver.ny)
                     for name,values in profiles.items()})
        np.savez_compressed(output/'trajectory.npz',**data)
    return data
