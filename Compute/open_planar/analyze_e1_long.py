"""Read-only E1 mean-thermal assessment; protocol fixed in e1_long_study.yaml.

Endpoint states are integrated piecewise linearly. Exchange energies are
interval sums, never trapezoid-integrated again. All reported gas storage is
independent of the membrane ODE. Run from the project root with -m.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from config_io import load_config
from core.unit_mapping import create_unit_mapping
from postproc.power import energy_balances
from simulation.output import dump


PROTOCOL = dict(window_cycles=10,
    temperature_absolute_K=.01, temperature_excess_relative=.01,
    velocity_absolute_m_s=1e-5, velocity_relative=.01,
    outlet_density_pressure_relative=1e-4, storage_power_relative=.01,
    power_change_relative=.01)


def assessment_protocol(requested_cycles):
    """Fix the last four full startup-aligned windows from the run request."""
    width=PROTOCOL['window_cycles']
    end=int(np.floor(float(requested_cycles)/width+1e-10))*width
    windows=[[start,start+width] for start in range(end-4*width,end,width)] if end>=4*width else []
    return dict(PROTOCOL,assessment_windows=windows)


def mean_state(t, x, start, end):
    """Exact integral of the saved piecewise-linear endpoint representation."""
    t, x = np.asarray(t), np.asarray(x)
    if not (np.all(np.diff(t)>0) and t[0]<=start+1e-14 and t[-1]>=end-1e-14):
        raise ValueError('ordered state samples must cover the full window')
    def at(v):
        j = int(np.clip(np.searchsorted(t,v,side='right')-1,0,len(t)-2))
        return x[j]+(x[j+1]-x[j])*(v-t[j])/(t[j+1]-t[j])
    lo,hi=np.searchsorted(t,[start,end],side='right')
    if hi and abs(t[hi-1]-end)<1e-14:
        hi-=1
    times=np.r_[start,t[lo:hi],end]
    values=np.concatenate((np.asarray(at(start))[None],x[lo:hi],np.asarray(at(end))[None]))
    return np.trapezoid(values,times,axis=0)/(end-start)


def inspect_case(case):
    config=load_config(case/'effective.yaml')
    summary=json.loads((case/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((case/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':
        raise ValueError('formal mean-thermal analysis requires a completed run')
    with np.load(case/'trajectory.npz') as raw:
        data={k:raw[k] for k in raw.files}
    if not bool(data['run_completed']):
        raise ValueError('incomplete trajectory')
    if not all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fiu'):
        raise ValueError('nonfinite saved numerical data')
    return data,config,summary,manifest


def assess(data,config,summary,manifest):
    run=config['run']; m=create_unit_mapping(config)
    protocol=assessment_protocol(run['cycles'])
    f=run['frequency_Hz']; dt=m.lattice.dt_s; area=run['power']['area_m2']
    pin=run['power']['mean_W']; T0=m.physical.T0_K
    p0=summary['air']['p0_Pa']; rho0=m.physical.rho0_kg_m3
    t=np.r_[0.,data['t_s']]; tp=data['profile_t_s']
    ncycles=int(np.floor(data['t_s'][-1]*f+1e-8))
    E0=float(data['initial_gas_energy_J_m2'])
    states=dict(film_temperature_K=np.r_[float(data['initial_film_temperature_K']),data['film_temperature_K']],
        gas_energy_J_m2=np.r_[E0,data['gas_energy_J_m2']],
        outlet_temperature_K=np.r_[T0,data['outlet_temperature_K']],
        outlet_density_kg_m3=np.r_[rho0,data['outlet_density_kg_m3']],
        outlet_velocity_m_s=np.r_[0.,data['outlet_velocity_m_s']],
        outlet_pressure_Pa=np.r_[p0,data['outlet_pressure_Pa']])
    profile_keys=('temperature_K','density_kg_m3','velocity_m_s','pressure_Pa')
    # SI thermodynamic identity, using the public mapping's EOS and dx.
    cv=m.physical.cp_J_kgK/m.physical.gamma
    U=np.sum(data['profile_pressure_Pa'],axis=1)*m.lattice.dx_m/(m.physical.gamma-1)
    K=np.sum(.5*data['profile_density_kg_m3']*data['profile_velocity_m_s']**2,axis=1)*m.lattice.dx_m
    sensible=np.sum(data['profile_density_kg_m3']*cv*(data['profile_temperature_K']-T0),axis=1)*m.lattice.dx_m
    mass=np.sum(data['profile_density_kg_m3'],axis=1)*m.lattice.dx_m
    profile_energy_error=float(np.max(abs(U+K-np.interp(tp,t,states['gas_energy_J_m2']))))
    windows=[]; profiles=[]
    for start in range(0,ncycles-9,10):
        end=start+10; a=int(round(start/f/dt)); b=int(round(end/f/dt)); duration=(b-a)*dt
        if abs(t[a]-start/f)>dt*1e-4 or abs(t[b]-end/f)>dt*1e-4:
            raise ValueError('window boundaries must coincide with saved steps for interval sums')
        row=dict(start_cycle=start,end_cycle=end,mid_time_ms=(start+end)/2/f*1e3)
        row.update({k:float(mean_state(t,v,start/f,end/f)) for k,v in states.items()})
        row['gas_energy_change_from_cold_J_m2']=row['gas_energy_J_m2']-E0
        prof={k:mean_state(tp,data['profile_'+k],start/f,end/f) for k in profile_keys}
        profiles.append(prof)
        row.update(mean_gas_temperature_K=float(prof['temperature_K'].mean()),
            max_mean_velocity_m_s=float(abs(prof['velocity_m_s']).max()),
            gas_mass_kg_m2=float(mean_state(tp,mass,start/f,end/f)),
            gas_sensible_excess_J_m2=float(mean_state(tp,sensible,start/f,end/f)))
        row.update(input_W=float(data['input_energy_J_m2'][a:b].sum()*area/duration),
            two_side_wall_W=float(2*data['audit'][a:b,2].sum()*area/duration),
            leak_W=float(data['leak_energy_J_m2'][a:b].sum()*area/duration),
            two_side_outlet_export_W=float(-2*data['audit'][a:b,3].sum()*area/duration),
            film_storage_rate_W=float(run['film']['C_A_J_m2K']*(states['film_temperature_K'][b]-states['film_temperature_K'][a])*area/duration),
            two_side_gas_storage_rate_W=float(2*(states['gas_energy_J_m2'][b]-states['gas_energy_J_m2'][a])*area/duration))
        row['balance_residual_W']=(row['film_storage_rate_W']+row['two_side_gas_storage_rate_W']
            -row['input_W']+row['leak_W']+row['two_side_outlet_export_W'])
        windows.append(row)
    transitions=[]
    for j in range(1,len(windows)):
        now,old=windows[j],windows[j-1]; newp,oldp=profiles[j],profiles[j-1]
        dT=abs(now['film_temperature_K']-old['film_temperature_K'])
        dprofile=float(abs(newp['temperature_K']-oldp['temperature_K']).max())
        du=float(abs(newp['velocity_m_s']-oldp['velocity_m_s']).max())
        # Denominators are heated-state excess, not 300 K; unresolved zero fails.
        film_limit=min(.01,.01*abs(now['film_temperature_K']-T0))
        profile_limit=min(.01,.01*float(abs(newp['temperature_K']-T0).max()))
        velocity_limit=max(1e-5,.01*float(abs(newp['velocity_m_s']).max()))
        power_change=max(abs(now[k]-old[k]) for k in ('two_side_wall_W','leak_W','two_side_outlet_export_W'))
        density_change=abs(now['outlet_density_kg_m3']-old['outlet_density_kg_m3'])/rho0
        pressure_change=abs(now['outlet_pressure_Pa']-old['outlet_pressure_Pa'])/p0
        gates=dict(film_temperature=dT<film_limit,temperature_profile=dprofile<profile_limit,
            mean_velocity=du<velocity_limit,power_change=power_change<.01*pin,
            outlet_density=density_change<1e-4,outlet_pressure=pressure_change<1e-4)
        transitions.append(dict(previous_window=[old['start_cycle'],old['end_cycle']],
            current_window=[now['start_cycle'],now['end_cycle']],film_difference_K=dT,
            film_limit_K=film_limit,profile_max_difference_K=dprofile,profile_limit_K=profile_limit,
            velocity_max_difference_m_s=du,velocity_limit_m_s=velocity_limit,
            max_power_difference_W=power_change,outlet_density_relative_difference=density_change,
            outlet_pressure_relative_difference=pressure_change,gates=gates,passed=all(gates.values())))
    selected=[j for j,w in enumerate(windows) if [w['start_cycle'],w['end_cycle']] in protocol['assessment_windows']]
    eligible=len(selected)==4
    storage_checks=[dict(window=[windows[j]['start_cycle'],windows[j]['end_cycle']],
        film=abs(windows[j]['film_storage_rate_W'])<.01*pin,
        gas=abs(windows[j]['two_side_gas_storage_rate_W'])<.01*pin) for j in selected]
    selected_transitions=[transitions[j-1] for j in selected[1:]]
    numeric_gates={k:v for k,v in summary['gates'].items() if k!='periodic_response'}
    passed=bool(eligible and all(x['passed'] for x in selected_transitions)
        and all(x['film'] and x['gas'] for x in storage_checks) and all(numeric_gates.values()))
    b=energy_balances(data,run['film'])
    energy={k:float(v*area) for k,v in summary['energy_final_J_m2'].items()}
    energy['two_side_gas_storage']=2*energy['one_side_gas_storage']
    energy['two_side_wall']=2*energy['one_side_wall']
    energy['two_side_outlet_export']=-2*energy['one_side_outlet_into_gas']
    energy['total_independent_balance_residual']=float(b['total_cumulative'][-1]*area)
    result=dict(status=manifest['status'],validation_passed=summary['validation_passed'],
        periodic_response_stable=summary['periodic_response_stable'],
        mean_thermal_assessed=eligible,mean_thermal_stable=passed if eligible else None,
        observation_protocol=protocol,storage_checks=storage_checks,
        assessment_transitions=selected_transitions,numerical_gates=numeric_gates,
        all_window_transitions=transitions,energy_J=energy,
        source=dict(run_git_commit=manifest['git_commit'],git_dirty_at_start=manifest['git_dirty'],
            source_clean_at_start=manifest['source_clean'],source_id=manifest['source_id']),
        independent_profile_total_energy_max_error_J_m2=profile_energy_error,
        gas_storage_definition='Eulerian total internal + kinetic energy; includes changing gas mass',
        gas_sensible_definition='Integral rho cv (T-T0) dz; warming diagnostic, not the conserved storage',
        profile_integration='All 32 phase samples per cycle, piecewise-linear time integral',
        envelope_definition='Per-cycle exact sample minimum/maximum; not uncertainty bands',
        outlet_maximum_changes=summary['outlet'],
        limits=['One r4 grid and about 1 mm domain only','No finite-film radiation validation',
            'Cold-reference linear normal-incidence outlet; warmed-background reflection not independently tested',
            'Finite observation windows cannot establish infinite-time equilibrium'])
    if windows:
        result['last_window']=windows[-1]
        result['outlet_last_window_relative_to_cold']=dict(
            temperature_K=windows[-1]['outlet_temperature_K']-T0,
            density_fraction=windows[-1]['outlet_density_kg_m3']/rho0-1,
            velocity_m_s=windows[-1]['outlet_velocity_m_s'],
            mach=windows[-1]['outlet_velocity_m_s']/m.physical.c0_m_s,
            pressure_Pa=windows[-1]['outlet_pressure_Pa']-p0)
    cycles=[]
    for cycle in range(ncycles):
        a=int(round(cycle/f/dt)); z=int(round((cycle+1)/f/dt))
        values=states['film_temperature_K'][a:z+1]
        cycles.append(dict(cycle=cycle+1,time_ms=(cycle+.5)/f*1e3,
            film_mean_K=float(np.trapezoid(values,dx=dt)/(1/f)),film_min_K=float(values.min()),film_max_K=float(values.max()),
            pressure_min_Pa=data['pressure_Pa'][a:z].min(axis=0),
            pressure_max_Pa=data['pressure_Pa'][a:z].max(axis=0)))
    return result,windows,profiles,cycles,b


def write_tables(output,result,windows):
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'mean_thermal_summary.json',result)
    if windows:
        with (output/'thermal_windows.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(windows[0]))
            writer.writeheader(); writer.writerows(windows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(); case=args.case_dir.resolve()
    output=args.output_dir or case.parent.parent/'final'
    if (output/'mean_thermal_summary.json').exists():
        raise FileExistsError('analysis exists; use a new output directory')
    data,config,summary,manifest=inspect_case(case)
    result,windows,profiles,cycles,balances=assess(data,config,summary,manifest)
    result['source'].update(analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_tables(output,result,windows)
    print(json.dumps({k:result[k] for k in ('status','validation_passed','periodic_response_stable',
        'mean_thermal_assessed','mean_thermal_stable','energy_J')},indent=2))


if __name__=='__main__':
    main()
