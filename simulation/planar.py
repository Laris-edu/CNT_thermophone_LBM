"""Reusable planar experiments. Physics remains in core/boundary; analysis in postproc."""
from dataclasses import asdict
import time

import numpy as np

from config_io import load_config
from core.planar_open_solver import PlanarOpenSolver
from postproc.planar_open import analyze_pulse, analyze_thermal
from reference.lim2013 import matched_gas
from simulation.study import BASE, DEFAULTS, merge


def make_solver(refine=1, length=1e-3):
    """Legacy construction helper, also usable for direct step-level experiments."""
    if refine < 1 or not np.isfinite(length) or length <= 0:
        raise ValueError('refine>=1 and finite length>0 required')
    config=load_config(BASE)
    config['lattice']['dx_m'] /= refine
    config['lattice']['dt_s'] /= refine
    config['numerics'].update(nx=1,ny=int(round(length/config['lattice']['dx_m'])),boundary='planar_open_y')
    config['case'].update(name='planar_open_prescribed_temperature',purpose='单侧给温开放域直接压力验证')
    return PlanarOpenSolver(config)


def collect_pulse(solver, output, run):
    m=solver.mapping; gas=matched_gas(m); pulse=run['pulse']
    H=solver.ny*m.lattice.dx_m; z=solver.z_m
    epsilon=pulse['epsilon']*np.exp(-.5*((z-pulse['center_fraction']*H)/(pulse['width_fraction']*H))**2)
    rho=m.lattice.rho_ref_lu*(1+epsilon[:,None])
    theta=m.theta_ref_lu*(1+epsilon[:,None])**(gas.gamma-1)
    u=np.zeros((solver.ny,1,2)); u[:,0,1]=m.c0_lu*epsilon
    solver.initialize_from_macro(rho,u,theta)
    row=int(np.argmin(abs(z-pulse['probe_fraction']*H)))
    steps=int(np.ceil(pulse['duration_crossings']*H/gas.c0_m_s/m.lattice.dt_s))
    history,audits=[],[]; completed=False; started=time.perf_counter()
    try:
        for _ in range(steps):
            audit=solver.step_temperature(gas.T0_K)
            state=solver.get_macro()
            dp=state.p[row,0]*m.pressure_scale-gas.p0_Pa
            rhocu=gas.rho0_kg_m3*gas.c0_m_s*state.u[row,0,1]*m.velocity_scale
            history.append([solver.t_lu*m.lattice.dt_s,dp,.5*(dp+rhocu),.5*(dp-rhocu)])
            audits.append(list(asdict(audit).values()))
        completed=True
    finally:
        h=np.asarray(history).reshape(-1,4)
        data=dict(t_s=h[:,0],pressure_Pa=h[:,1],outgoing_Pa=h[:,2],incoming_Pa=h[:,3],
            audit=np.asarray(audits).reshape(-1,6),probe_z_m=z[row],actual_length_m=H,
            run_completed=completed,elapsed_s=time.perf_counter()-started)
        np.savez_compressed(output/'trajectory.npz',**data)
    return data


def collect_thermal(solver, output, run):
    m=solver.mapping; gas=matched_gas(m); dt=m.lattice.dt_s
    frequency=run['frequency_Hz']; amplitude=run['amplitude_K']; cycles=run['cycles']; ramp_cycles=run['ramp_cycles']
    requested=np.asarray(run['probes_m']); rows=np.array([np.argmin(abs(solver.z_m-z)) for z in requested])
    steps=int(np.ceil(cycles/frequency/dt)); count=len(rows)
    history,audits=[],[]; started=time.perf_counter(); last_cycle=0; completed=False
    initial=solver.get_macro(); initial_mass=float(np.sum(initial.rho))
    initial_energy=float(np.sum(initial.E_tot)*m.pressure_scale*m.lattice.dx_m)
    min_f=min_g=float('inf'); profile_t,profile_T,profile_p,profile_u=[],[],[],[]
    velocities=[]
    try:
        for i in range(steps):
            tq=(i+.5)*dt
            ramp=.5*(1-np.cos(np.pi*min(1.,tq*frequency/ramp_cycles))) if ramp_cycles else 1.
            Tw=gas.T0_K+amplitude*ramp*np.sin(2*np.pi*frequency*tq)
            audit=solver.step_temperature(Tw); state=solver.get_macro()
            min_f=min(min_f,float(np.min(solver.f))); min_g=min(min_g,float(np.min(solver.g)))
            history.append([(i+1)*dt,tq,Tw,audit.wall_energy_J_m2/dt,*state.p[rows,0]*m.pressure_scale,
                state.theta[0,0]*m.temperature_scale,float(np.sum(state.rho)),
                float(np.sum(state.E_tot)*m.pressure_scale*m.lattice.dx_m)])
            velocities.append(state.u[rows,0,1]*m.velocity_scale)
            audits.append(list(asdict(audit).values()))
            if ((i+1)*dt*frequency >= cycles-run['sampling']['profile_cycles']
                and i % max(1,int(1/(frequency*dt*run['sampling']['profile_samples_per_cycle']))) == 0):
                profile_t.append((i+1)*dt)
                profile_T.append(state.theta[:,0]*m.temperature_scale)
                profile_p.append(state.p[:,0]*m.pressure_scale)
                profile_u.append(state.u[:,0,1]*m.velocity_scale)
            current=int((i+1)*dt*frequency)
            if current>last_cycle:
                print(f'  cycle {current}/{cycles:g}, {time.perf_counter()-started:.1f} s',flush=True)
                last_cycle=current
        completed=True
    finally:
        h=np.asarray(history).reshape(-1,7+count)
        data=dict(t_s=h[:,0],heat_time_s=h[:,1],wall_temperature_K=h[:,2],wall_flux_W_m2=h[:,3],
            pressure_Pa=h[:,4:4+count],first_cell_temperature_K=h[:,4+count],mass_lu=h[:,5+count],
            gas_energy_J_m2=h[:,6+count],audit=np.asarray(audits).reshape(-1,6),probe_z_m=solver.z_m[rows],
            initial_mass_lu=initial_mass,initial_gas_energy_J_m2=initial_energy,z_m=solver.z_m,
            profile_t_s=np.asarray(profile_t),profile_temperature_K=np.asarray(profile_T).reshape(-1,solver.ny),
            profile_pressure_Pa=np.asarray(profile_p).reshape(-1,solver.ny),min_f=min_f,min_g=min_g,
            velocity_m_s=np.asarray(velocities).reshape(-1,count),
            profile_velocity_m_s=np.asarray(profile_u).reshape(-1,solver.ny),
            run_completed=completed,elapsed_s=time.perf_counter()-started)
        np.savez_compressed(output/'trajectory.npz',**data)
    return data


def pulse_case(solver, output):
    """Compatibility helper; writes only the trajectory, as the old function did."""
    run=merge(DEFAULTS,{'mode':'pulse'})
    return analyze_pulse(collect_pulse(solver,output,run),solver.mapping,run)


def thermal_case(solver, output, frequency, amplitude, cycles, ramp_cycles):
    run=merge(DEFAULTS,dict(frequency_Hz=frequency,amplitude_K=amplitude,cycles=cycles,ramp_cycles=ramp_cycles))
    if solver.z_m[-1] <= max(run['probes_m']):
        raise ValueError('domain must contain all requested probes')
    return analyze_thermal(collect_thermal(solver,output,run),solver.mapping,run)
