"""Finite strip case lifecycle, with separate numerical and boundary gates.

The experimental 2D exit is never certified by conservation alone. Pulse
comparison needs an independently advanced larger domain at the same grid.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import math
import time

import numpy as np
import yaml

from config_io import load_config
from core.unit_mapping import create_unit_mapping
from simulation.output import dump
from simulation.provenance import capture_source
from simulation.study import ROOT, valid_name


DEFAULTS=dict(kind='power',dx_m=1.305875e-6,width_m=.002,height_m=.0005,
    film_width_m=.001,frequency_Hz=20000.,cycles=24.,duration_s=8e-6,
    mean_power_W_m2=1.13/9e-4,C_A_J_m2K=.012888,sheet_conductance_W_K=5.7996e-7,
    G_W_m2K=0.,initial_temperature_K=300.,sponge_width_m=.0001,sponge_rate=.05,
    pulse_amplitude_Pa=1.,pulse_sigma_m=75e-6,probe_radius_m=.00025,
    sample_every=1,field_samples_per_cycle=32,stop_when_stable=True,
    pml_width_m=0.,pml_target_return=.001,pml_alpha_Hz=1000.)


def prepare_strip_case(parameters, base=None):
    p={**DEFAULTS,**deepcopy(parameters)}
    name=valid_name(p.get('name','finite_strip'))
    if p['kind'] not in ('power','pulse'):
        raise ValueError('strip kind must be power or pulse')
    if p['kind']=='pulse' and 'film_width_m' not in parameters:
        p['film_width_m']=0.
    if p['kind']=='pulse' and p['film_width_m']!=0:
        raise ValueError('boundary pulse uses no film; it does not integrate a finite-capacity film')
    for key in ('dx_m','width_m','height_m','frequency_Hz','cycles','duration_s',
                'C_A_J_m2K','initial_temperature_K','pulse_sigma_m','probe_radius_m'):
        if not np.isfinite(p[key]) or p[key]<=0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('film_width_m','sheet_conductance_W_K','G_W_m2K','sponge_width_m','sponge_rate',
                'pml_width_m','pml_alpha_Hz'):
        if not np.isfinite(p[key]) or p[key]<0:
            raise ValueError(f'{key} must be nonnegative and finite')
    for key in ('sample_every','field_samples_per_cycle'):
        if type(p[key]) is not int or p[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if p['film_width_m']>=p['width_m'] or (p['kind']=='power' and p['film_width_m']==0):
        raise ValueError('finite film must have positive side margins')
    if not np.isfinite(p['pml_target_return']) or not 0<p['pml_target_return']<1:
        raise ValueError('pml_target_return must lie in (0,1)')
    if p['pml_width_m'] and p['sponge_rate']:
        raise ValueError('PML and legacy sponge cannot be combined')
    c=load_config(Path(base) if base else ROOT/'Compute/open_planar/e1_long_air.yaml')
    scale=p['dx_m']/c['lattice']['dx_m']
    c['lattice']['dt_s']*=scale
    c['lattice']['dx_m']=p['dx_m']
    nx=int(round(p['width_m']/p['dx_m'])); ny=int(round(p['height_m']/p['dx_m']))
    if min(nx,ny)<8:
        raise ValueError('finite strip requires at least eight cells per axis')
    c['numerics'].update(nx=nx,ny=ny,collision_backend='numba',boundary='finite_strip_open_xy')
    c['case']={'name':name,'purpose':'finite-width infinite strip, symmetric two-sided heating'}
    m=create_unit_mapping(c)
    duration=p['cycles']/p['frequency_Hz'] if p['kind']=='power' else p['duration_s']
    steps=int(math.ceil(duration/m.lattice.dt_s-1e-9))
    p.update(requested_width_m=p['width_m'],requested_height_m=p['height_m'],
             actual_width_m=nx*p['dx_m'],actual_height_m=ny*p['dx_m'],steps=steps,
             actual_duration_s=steps*m.lattice.dt_s,
             sponge_cells=int(round(p['sponge_width_m']/p['dx_m'])),
             pml_cells=int(round(p['pml_width_m']/p['dx_m'])))
    c['strip']=p
    return c


def _totals(solver):
    f,g=solver._require_state()
    c2=np.sum(solver.lattice.c**2,axis=1)
    return float(np.sum(f)),float(np.sum(.5*f*c2+g)*solver.mapping.pressure_scale*solver.mapping.lattice.dx_m**2)


def _sample_bilinear(values,x,y,px,py):
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((y,x),values,bounds_error=True)(np.column_stack((py,px)))


def _pulse_initialize(solver,run):
    m=solver.mapping
    x,y=np.meshgrid(solver.x_m,solver.y_m)
    pressure0=m.lattice.rho_ref_lu*m.theta_ref_lu*m.pressure_scale
    ratio=1+run['pulse_amplitude_Pa']/pressure0*np.exp(-.5*(x*x+y*y)/run['pulse_sigma_m']**2)
    rho=m.lattice.rho_ref_lu*ratio**(1/m.gamma_from_degrees)
    theta=m.theta_ref_lu*ratio**(1-1/m.gamma_from_degrees)
    solver.initialize_from_macro(rho,np.zeros((solver.ny,solver.nx,2)),theta)


def collect_strip(solver,run):
    """Collect endpoint fields and interval-integrated exchanges, without clipping."""
    m=solver.mapping; dt=m.lattice.dt_s
    if run['kind']=='pulse':
        _pulse_initialize(solver,run)
    else:
        solver._require_state()
    initial_mass,initial_energy=_totals(solver)
    theta0=m.theta_ref_lu; rho0=m.lattice.rho_ref_lu
    pressure0=rho0*theta0*m.pressure_scale
    # The two endpoints are near the symmetry plane, not on an extrapolated row.
    angles=np.linspace(.04,np.pi-.04,13)
    px=run['probe_radius_m']*np.cos(angles)
    py=run['probe_radius_m']*np.sin(angles)
    py=np.maximum(py,solver.y_m[0])
    if run['kind']=='power':
        # Exterior to all three handoff contours, but inside the active gas.
        px=np.r_[px,0.,-.00078,.00078]
        py=np.r_[py,.00028,.0001,.0001]
    records={key:[] for key in ('t_s','pressure_perturbation_Pa','velocity_x_m_s',
        'velocity_y_m_s','gas_mass_lu','gas_energy_J_m','audit','film_temperature_K',
        'film_wall_energy_J_m2','film_input_energy_J_m','film_leak_energy_J_m',
        'audit_t_s','wall_energy_J_m','film_storage_J_m')}
    audit_columns=None
    temperature=np.full(len(solver.film_x_m),run['initial_temperature_K'])
    film_initial=temperature.copy()
    minimum_f=minimum_g=math.inf
    start=time.perf_counter(); stop_reason='maximum_steps'
    from coupling.film_ode import FilmOdeParams
    params=FilmOdeParams(C_A_si=run['C_A_J_m2K'],T_ref_K=run['initial_temperature_K'],
        linear_leak_conductance_si=run['G_W_m2K'],gas_flux_factor=2.)
    power_integral=leak_integral=0.
    max_film_residual=0.
    max_wall_mass=0.
    surface_records=[]; surface_times=[]; surfaces=[]; surface_slices=[]
    stationary_history=[]
    if run['kind']=='power':
        from farfield.strip_surface import upper_rectangle
        # Each contour is wholly inside the gas domain and outside the film.
        for half_width,height in ((.00065,.00015),(.0007,.0002),(.00075,.00025)):
            if half_width>=max(solver.x_m) or height>=max(solver.y_m):
                raise ValueError('power domain does not contain all three handoff contours')
            margin=max(run['sponge_cells'],run.get('pml_cells',0))*m.lattice.dx_m
            if (half_width<=max(abs(solver.film_x_min_m),abs(solver.film_x_max_m))
                    or half_width>=.5*solver.nx*m.lattice.dx_m-margin
                    or height>=solver.ny*m.lattice.dx_m-margin):
                raise ValueError('handoff contours must enclose film and stay outside sponge/PML')
            surfaces.append(upper_rectangle(half_width,height,2*m.lattice.dx_m))
        count=0
        for surface in surfaces:
            surface_slices.append((count,count+len(surface['x_m'])))
            count=surface_slices[-1][1]
        surface_all={k:np.concatenate([s[k] for s in surfaces]) for k in surfaces[0]}
    field_stride=max(1,int(round(1/(run['frequency_Hz']*dt*run['field_samples_per_cycle']))))
    next_cycle=8
    log_interval=max(1,min(100,run['steps']//20))
    for step in range(run['steps']):
        midpoint=(step+.5)*dt
        if run['kind']=='pulse':
            audit=solver.step_temperature(run['initial_temperature_K'])
            work=leak=0.
        else:
            from coupling.finite_strip import solve_strip_film_midpoint, audit_strip_film_step
            power=run['mean_power_W_m2']*(1-np.cos(2*np.pi*run['frequency_Hz']*midpoint))
            trial=[]
            def solve(exchange):
                root=solve_strip_film_midpoint(params=params,T_s_K=temperature,
                    dx_m=solver.film_dx_m,sheet_conductance_W_K=run['sheet_conductance_W_K'],
                    dt_s=dt,power_W_m2=power,exchange_J_m2=exchange,exchange_is_local=True)
                trial.append(root)
                return root
            audit=solver.step_wall_energy(solve)
            new=2*trial[-1]-temperature
            wall=np.asarray(audit.wall_energy_J_m2)
            film_audit=audit_strip_film_step(params=params,T_old_K=temperature,T_new_K=new,
                dx_m=solver.film_dx_m,dt_s=dt,power_W_m2=power,wall_energy_J_m2=wall)
            work,leak=film_audit.input_energy_J_m,film_audit.leak_energy_J_m
            max_film_residual=max(max_film_residual,abs(film_audit.residual_J_m))
            temperature=new
        power_integral+=work; leak_integral+=leak
        # Reuse the independent actual-population sums made before the accepted
        # step was committed, never a value reconstructed from an equation RHS.
        mass,energy=audit.gas_mass_after_lu,audit.gas_energy_after_J_m
        minimum_f=min(minimum_f,audit.minimum_f); minimum_g=min(minimum_g,audit.minimum_g)
        item=asdict(audit)
        if np.size(audit.wall_mass_lu):
            max_wall_mass=max(max_wall_mass,float(np.max(abs(audit.wall_mass_lu))))
        scalar={k:float(v) for k,v in item.items() if np.ndim(v)==0 and isinstance(v,(float,int,np.number))}
        if audit_columns is None:
            audit_columns=list(scalar)
        # Keep audit and storage every step even when field observations are decimated.
        observed=((step+1)%(field_stride if surfaces else run['sample_every'])==0 or step+1==run['steps'])
        records['audit_t_s'].append((step+1)*dt)
        records['gas_mass_lu'].append(mass); records['gas_energy_J_m'].append(energy)
        records['audit'].append([scalar[k] for k in audit_columns])
        records['wall_energy_J_m'].append(float(np.sum(audit.wall_energy_J_m2)*solver.film_dx_m))
        records['film_storage_J_m'].append(float(np.sum(params.C_A_si*(temperature-film_initial))*solver.film_dx_m))
        if observed:
            macro=solver.get_macro()
            records['t_s'].append((step+1)*dt)
            records['pressure_perturbation_Pa'].append(_sample_bilinear(macro.p*m.pressure_scale-pressure0,solver.x_m,solver.y_m,px,py))
            records['velocity_x_m_s'].append(_sample_bilinear(macro.u[...,0]*m.velocity_scale,solver.x_m,solver.y_m,px,py))
            records['velocity_y_m_s'].append(_sample_bilinear(macro.u[...,1]*m.velocity_scale,solver.x_m,solver.y_m,px,py))
            records['film_temperature_K'].append(temperature.copy())
            records['film_wall_energy_J_m2'].append(np.asarray(audit.wall_energy_J_m2).copy())
        records['film_input_energy_J_m'].append(work); records['film_leak_energy_J_m'].append(leak)
        if surfaces and observed:
            surface_times.append((step+1)*dt)
            # p/T/u are endpoint observations. Heat exchanges retain midpoints.
            channels=(macro.p*m.pressure_scale-pressure0,macro.u[...,0]*m.velocity_scale,
                      macro.u[...,1]*m.velocity_scale,macro.theta*m.temperature_scale,
                      macro.rho*m.rho_scale)
            surface_records.append(np.stack([_sample_bilinear(v,solver.x_m,solver.y_m,
                surface_all['x_m'],surface_all['y_m']) for v in channels],axis=-1))
        if run['kind']=='power' and (step+1)*dt*run['frequency_Hz']>=next_cycle:
            from postproc.power import stationarity_at_cycle
            state={k:np.asarray(v) for k,v in records.items()}
            selected=sorted(set([0,len(temperature)//2,len(temperature)-1]))
            checks=[]
            for index in selected:
                observation=dict(t_s=state['t_s'],heat_time_s=state['t_s']-.5*dt,
                    film_temperature_K=state['film_temperature_K'][:,index],
                    wall_flux_W_m2=state['film_wall_energy_J_m2'][:,index]/dt,
                    pressure_Pa=state['pressure_perturbation_Pa'],velocity_m_s=state['velocity_y_m_s'])
                checks.append(stationarity_at_cycle(observation,run['frequency_Hz'],next_cycle,
                    dict(stationary_relative=.01,stationary_phase_deg=1.)))
            stable=all(v['passed'] for v in checks)
            full_check={'evaluated':False,'passed':False}
            if stable:
                # Sparse probes are a preliminary screen. The exported curve
                # and every membrane bin must also pass the SAME public fit
                # windows and thresholds before a long run can stop early.
                fields=np.asarray(surface_records)
                observation=dict(t_s=np.asarray(surface_times),heat_time_s=state['t_s']-.5*dt,
                    film_temperature_K=state['film_temperature_K'][:,0],
                    wall_flux_W_m2=state['film_wall_energy_J_m2'][:,0]/dt,
                    pressure_Pa=fields[...,0],velocity_m_s=fields[...,1]*surface_all['normal_x']
                        +fields[...,2]*surface_all['normal_y'])
                surface_check=stationarity_at_cycle(observation,run['frequency_Hz'],next_cycle,
                    dict(stationary_relative=.01,stationary_phase_deg=1.))
                failed_bins=[]
                for index in range(len(temperature)):
                    item=dict(t_s=state['t_s'],heat_time_s=state['t_s']-.5*dt,
                        film_temperature_K=state['film_temperature_K'][:,index],
                        wall_flux_W_m2=state['film_wall_energy_J_m2'][:,index]/dt,
                        pressure_Pa=np.empty((len(state['t_s']),0)),
                        velocity_m_s=np.empty((len(state['t_s']),0)))
                    if not stationarity_at_cycle(item,run['frequency_Hz'],next_cycle,
                        dict(stationary_relative=.01,stationary_phase_deg=1.))['passed']:
                        failed_bins.append(index)
                stable=bool(surface_check['passed'] and not failed_bins)
                full_check=dict(evaluated=True,passed=stable,
                    surface_passed=surface_check['passed'],surface_points=fields.shape[1],
                    film_bins=len(temperature),failed_film_bins=failed_bins,
                    failed_surface_channels=[name for name,value in surface_check.get('channels',{}).items()
                                             if not value['passed']])
            stationary_history.append(dict(cycle=next_cycle,passed=stable,film_bin_checks=checks,
                                             full_field_check=full_check))
            next_cycle+=1
            if run['stop_when_stable'] and stable:
                stop_reason='periodic_response_stable'
                break
        if (step+1)%log_interval==0:
            print(f'  {step+1}/{run["steps"]} steps, t={(step+1)*dt:.6g}s, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    # A stable-cycle stop need not fall on the regular observation stride.
    if not observed:
        macro=solver.get_macro()
    data={k:np.asarray(v) for k,v in records.items()}
    data.update(heat_time_s=data['t_s']-.5*dt,probe_x_m=px,probe_y_m=py,
        audit_columns=np.asarray(audit_columns),initial_gas_mass_lu=initial_mass,
        initial_gas_energy_J_m=initial_energy,initial_film_temperature_K=film_initial,
        film_x_m=solver.film_x_m,film_dx_m=solver.film_dx_m,x_m=solver.x_m,y_m=solver.y_m,
        final_pressure_perturbation_Pa=macro.p*m.pressure_scale-pressure0,
        final_temperature_K=macro.theta*m.temperature_scale,
        final_velocity_m_s=macro.u*m.velocity_scale,
        minimum_f=minimum_f,minimum_g=minimum_g,max_film_energy_residual_J_m=max_film_residual,
        max_wall_mass_lu=max_wall_mass,periodic_response_stable=bool(stationary_history and stationary_history[-1]['passed']),
        elapsed_s=time.perf_counter()-start,stop_reason=stop_reason,run_completed=True)
    if surfaces:
        import json
        data.update(surface_t_s=np.asarray(surface_times),surface_fields=np.asarray(surface_records),
            surface_channels=np.array(['pressure_perturbation_Pa','ux_m_s','uy_m_s','temperature_K','rho_kg_m3']),
            surface_slices=np.asarray(surface_slices),stationarity_json=json.dumps(stationary_history))
        data.update({f'surface_{k}':v for k,v in surface_all.items()})
    return data


def summarize_strip(data,config):
    names=list(data['audit_columns']); a=data['audit']
    maximum=lambda name:float(np.max(abs(a[:,names.index(name)])))
    column=lambda key:a[:,names.index(key)]
    wall=data['wall_energy_J_m']
    gas=data['gas_energy_J_m']-float(data['initial_gas_energy_J_m'])
    pml_energy=column('pml_energy_J_m') if 'pml_energy_J_m' in names else 0.
    pml_mass=column('pml_mass_lu') if 'pml_mass_lu' in names else 0.
    gas_residual=gas-np.cumsum(wall+column('outer_energy_J_m')+column('sponge_energy_J_m')+pml_energy)
    mass_residual=data['gas_mass_lu']-float(data['initial_gas_mass_lu'])-np.cumsum(
        column('outer_mass_lu')+column('sponge_mass_lu')+pml_mass)
    film=data['film_storage_J_m']
    total_residual=film+2*gas-np.cumsum(data['film_input_energy_J_m']-data['film_leak_energy_J_m']
        +2*(column('outer_energy_J_m')+column('sponge_energy_J_m')+pml_energy))
    checks=dict(finite=bool(all(np.all(np.isfinite(v)) for k,v in data.items()
                     if isinstance(v,np.ndarray) and np.issubdtype(v.dtype,np.number))),
        positive=bool(data['minimum_f']>0 and data['minimum_g']>0),
        mass_conservation=maximum('mass_residual_lu')<1e-8,
        energy_conservation=maximum('energy_residual_J_m')<1e-10,
        wall_mass_conservation=float(data['max_wall_mass_lu'])<1e-12,
        cumulative_gas_energy=float(np.max(abs(gas_residual)))<1e-8,
        cumulative_total_energy=float(np.max(abs(total_residual)))<1e-8,
        cumulative_mass=float(np.max(abs(mass_residual)))<1e-6)
    if config['strip']['kind']=='power':
        checks['film_energy_conservation']=float(data['max_film_energy_residual_J_m'])<1e-10
    return dict(status='completed',kind=config['strip']['kind'],state_checks=checks,
        state_validation_passed=all(checks.values()),validation_passed=None,
        boundary_validation_passed=None,propagation_validation_passed=None,
        periodic_response_stable=bool(data['periodic_response_stable']) if config['strip']['kind']=='power' else None,
        frequency_20kHz_validated=False,
        steps=len(data['audit_t_s']),elapsed_s=float(data['elapsed_s']),
        max_mass_residual_lu=maximum('mass_residual_lu'),
        max_energy_residual_J_m=maximum('energy_residual_J_m'),
        max_film_energy_residual_J_m=float(data['max_film_energy_residual_J_m']),
        film_energy_check_applicable=config['strip']['kind']=='power',
        max_cumulative_gas_energy_residual_J_m=float(np.max(abs(gas_residual))),
        max_cumulative_total_energy_residual_J_m=float(np.max(abs(total_residual))),
        max_cumulative_mass_residual_lu=float(np.max(abs(mass_residual))),
        minimum_f=float(data['minimum_f']),minimum_g=float(data['minimum_g']),
        scope='experimental finite strip; conservation is not open-boundary or device validation')


def run_strip_case(effective,output):
    """Run only a new directory and retain failed-run status under four-file contract."""
    from core.finite_strip_solver import FiniteStripSolver
    c=deepcopy(effective); run=c['strip']; output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'case already exists: {output}')
    source=capture_source(ROOT)
    solver=FiniteStripSolver(c,film_width_m=run['film_width_m'],
        sponge_cells=run['sponge_cells'],sponge_rate=run['sponge_rate'],
        pml_cells=run.get('pml_cells',0),pml_target_return=run.get('pml_target_return',.001),
        pml_alpha_Hz=run.get('pml_alpha_Hz',1000.))
    run.update(actual_film_x_min_m=float(solver.film_x_min_m),
               actual_film_x_max_m=float(solver.film_x_max_m))
    manifest=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
        finished_utc=None,**source,geometry='2D infinite strip, upper symmetric half-space',
        collision_backend='numba',phase_convention='exp(+i omega t)',
        pressure='direct gas pressure',film_edge='adiabatic, no electrodes/support solids',
        exit=solver.boundary_model,
        boundary_validation_passed=None)
    output.mkdir(parents=True,exist_ok=False)
    (output/'effective.yaml').write_text(yaml.safe_dump(c,allow_unicode=True),encoding='utf-8')
    dump(output/'manifest.json',manifest)
    np.savez_compressed(output/'trajectory.npz',t_s=np.array([]),run_completed=False)
    dump(output/'summary.json',dict(status='running',validation_passed=None))
    started=time.perf_counter()
    try:
        data=collect_strip(solver,run)
        np.savez_compressed(output/'trajectory.npz',**data)
        summary=summarize_strip(data,c)
        dump(output/'summary.json',summary)
        manifest['status']='completed'
        return summary
    except BaseException as exc:
        manifest['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        error=dict(type=type(exc).__name__,message=str(exc))
        manifest['error']=error
        # Preserve the last accepted state even if a later trial fails. Never
        # relabel it a completed trajectory or fabricate an accepted step.
        if solver.f is not None:
            state=solver.get_macro()
            np.savez_compressed(output/'trajectory.npz',t_s=np.array([solver.t_lu*solver.mapping.lattice.dt_s]),
                run_completed=False,accepted_steps=solver.t_lu,x_m=solver.x_m,y_m=solver.y_m,
                last_pressure_Pa=state.p*solver.mapping.pressure_scale,
                last_temperature_K=state.theta*solver.mapping.temperature_scale,
                last_velocity_m_s=state.u*solver.mapping.velocity_scale)
        dump(output/'summary.json',dict(status=manifest['status'],validation_passed=None,error=error))
        raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-started)
        dump(output/'manifest.json',manifest)


def compare_pulse_cases(trial,reference,*,return_start_s,limit=.01,trial_effective=None,reference_effective=None):
    """Return error against a larger LBM domain; not a plane-wave p+/p- metric."""
    for name in ('probe_x_m','probe_y_m','t_s'):
        if not np.allclose(trial[name],reference[name],rtol=0,atol=1e-14):
            raise ValueError('pulse comparison requires matching coordinates and endpoint times')
    ref=np.asarray(reference['pressure_perturbation_Pa'])
    actual=np.asarray(trial['pressure_perturbation_Pa'])
    mask=np.asarray(trial['t_s'])>=return_start_s
    if not mask.any():
        raise ValueError('no samples after expected return time')
    scale=np.max(abs(ref),axis=0)
    if np.any(scale<1e-12):
        raise ValueError('unresolved incident reference pulse')
    ratios=np.max(abs(actual[mask]-ref[mask]),axis=0)/scale
    reference_checked=False; reference_return_s=None
    if trial_effective is not None and reference_effective is not None:
        for key in ('physical','lattice','transport','collision'):
            if trial_effective[key]!=reference_effective[key]:
                raise ValueError(f'boundary reference differs in {key}')
        small,big=trial_effective['strip'],reference_effective['strip']
        for key in ('kind','film_width_m','pulse_sigma_m','pulse_amplitude_Pa','sample_every'):
            if small[key]!=big[key]:
                raise ValueError(f'boundary reference differs in {key}')
        if (big['kind']!='pulse' or big['film_width_m']!=0
                or big['actual_width_m']<=small['actual_width_m']
                or big['actual_height_m']<=small['actual_height_m']):
            raise ValueError('boundary reference must be a larger film-free pulse domain')
        distance=min(big['actual_width_m']/2,big['actual_height_m'])-max(
            big['sponge_cells'],big.get('pml_cells',0))*big['dx_m']
        reference_return_s=(2*distance-big['probe_radius_m']-6*big['pulse_sigma_m'])/create_unit_mapping(reference_effective).physical.c0_m_s
        if reference_return_s<=float(reference['t_s'][-1]):
            raise ValueError('reference acoustic return can reach probes in this window')
        reference_checked=True
    agreement=bool(np.max(ratios)<limit)
    return dict(boundary_validation_passed=bool(agreement and reference_checked),
        pulse_agreement_passed=agreement,reference_configuration_checked=reference_checked,
        estimated_reference_acoustic_return_s=reference_return_s,
        max_return_error_ratio=float(np.max(ratios)),return_error_ratios=ratios.tolist(),
        limit=limit,return_start_s=return_start_s,frequency_20kHz_validated=False,
        metric='max absolute small-minus-large-domain pressure after return / peak reference pressure per probe',
        scope='broadband pulse at saved grid; not a 20 kHz harmonic or thermoviscous-resolution certificate')
