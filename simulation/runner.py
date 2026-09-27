"""Four-file case lifecycle and sequential study runner."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from config_io import load_config
from core.planar_open_solver import PlanarOpenSolver
from core.unit_mapping import create_unit_mapping
from postproc.planar_open import analyze_pulse, analyze_thermal
from postproc.power import analyze_power
from simulation.output import dump
from simulation.planar import collect_pulse, collect_thermal
from simulation.power import collect_power
from simulation.provenance import capture_source
from simulation.study import ROOT, DEFAULTS, load_study, merge, valid_name, validate_effective, synchronized_run


def _now():
    return datetime.now(timezone.utc).isoformat()


def _summary(data, config, previous=None):
    mapping=create_unit_mapping(config)
    run=config['run']
    if run['mode'] in ('interface_pulse','thermal_step'):
        from postproc.interface import analyze_interface_pulse, analyze_thermal_step
        analyze={'interface_pulse':analyze_interface_pulse,'thermal_step':analyze_thermal_step}[run['mode']]
    else:
        analyze={'pulse':analyze_pulse,'thermal':analyze_thermal,'power':analyze_power}[run['mode']]
    result=analyze(data,mapping,run,previous)
    if run['mode']=='power':
        result['model_assumptions']=deepcopy(config.get('case',{}))
        result['paper_air_comparison']={key:dict(paper=value,model=result['air'][key],
            difference_percent=100*(result['air'][key]/value-1))
            for key,value in config.get('case',{}).get('paper_air',{}).items()}
    result.update(mapping=mapping.to_metadata(),steps=len(data['t_s']),
        actual_length_m=config['numerics']['ny']*mapping.lattice.dx_m,
        validation_passed=bool(result['pass_reflection'] if run['mode'] in ('pulse','interface_pulse') else all(result['gates'].values())))
    if synchronized_run(run):
        result.update(sync_samples=len(data['t_s']),fine_steps=int(data['sync_fine_step'][-1]),
            level_steps=np.asarray(data['level_steps']).tolist(),cell_updates=int(data['cell_updates']),
            warmup_s=float(data['warmup_s']),advance_s=float(data['advance_s']),
            mass_lu_definition='SI areal mass / (rho_scale * finest dx)',
            grid=config.get('grid',{'type':'uniform'}),comparison_passed=None)
        if bool(data.get('interface_exchange_available',False)):
            result['max_interface_cancellation_SI']=np.max(abs(data['interface_exchange'][:,2,:]),axis=0).tolist()
    return result


def run_case(effective, output, *, source_cache=None, provenance=None):
    """Run into a NEW directory. The retired source_cache argument is ignored."""
    config=deepcopy(effective)
    validate_effective(config)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'case already exists: {output}')
    # Provenance is prepared before creating the case directory; it is shared per batch.
    source=deepcopy(provenance if provenance is not None else
        capture_source(ROOT))
    for name,digest in source['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('source changed during the batch; use a new process/batch')
    run=config['run']; started=time.perf_counter()
    manifest=dict(created_utc=_now(),finished_utc=None,status='running',elapsed_s=None,
        python=platform.python_version(),numpy=np.__version__,pyyaml=yaml.__version__,
        **source,
        args=dict(mode=run['mode'],refine=run['refine'],length=run['length_m'],frequency=run['frequency_Hz'],
            amplitude=run.get('amplitude_K'),cycles=run['cycles'],ramp_cycles=run.get('ramp_cycles'),output=str(output)),
        pressure='direct rho*theta; no heat-flux conversion',
        geometry='nx=1 planar upper half-space; z=(row+0.5)dx; open top; diffuse thermal bottom',
        audit_columns=['wall_mass_lu','top_mass_lu','wall_energy_J_m2','top_energy_J_m2',
            'mass_residual_lu','energy_residual_J_m2'])
    manifest['collision_backend'] = config['numerics'].get('collision_backend', 'numpy')
    if manifest['collision_backend'] == 'numba':
        manifest.update(numba=version('numba'), llvmlite=version('llvmlite'))
    if run['mode']=='power':
        manifest.update(power_drive='P_mean*(1-cos(2*pi*f_power*t)); electrical frequency=f_power/2',
                        power=run['power'],film=run['film'],stopping=run['stopping'],
                        heat_flux='actual interval wall energy / dt; positive film to upper gas',
                        phase_convention='exp(+i*omega*t); power harmonic is -P_mean/area')
    output.mkdir(parents=True,exist_ok=False)
    try:
        (output/'effective.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
        manifest['effective_sha256']=hashlib.sha256((output/'effective.yaml').read_bytes()).hexdigest()
        dump(output/'manifest.json',manifest)
        np.savez_compressed(output/'trajectory.npz',t_s=np.array([]),run_completed=False)
        dump(output/'summary.json',dict(status='running',validation_passed=None))
        if 'grid' in config:
            from core.two_level_planar_solver import TwoLevelPlanarOpenSolver
            solver=TwoLevelPlanarOpenSolver(config)
        else:
            solver=PlanarOpenSolver(config)
        print(f"{config['case']['name']}: {run['mode']}, ny={solver.ny}, dx={solver.mapping.lattice.dx_m:g}",flush=True)
        if synchronized_run(run):
            from simulation.multigrid import collect_synchronized
            collect=collect_synchronized
            manifest.update(mass_lu_definition='SI areal mass / (rho_scale * finest dx)',
                geometry='nx=1 planar physical cells at saved z_m with cell_width_m; shadows excluded',
                gas_time_axis='t_s: synchronized gas endpoints',
                film_time_axis='film_t_s: finest-step film endpoints',
                heat_time_axis='heat_time_s: finest-step interval midpoints',
                interface_exchange_columns=['mass_kg_m2','momentum_x_kg_m_s','momentum_y_kg_m_s','energy_J_m2'],
                interface_exchange_rows=['into_lower','into_upper','sum'],
                grid_layout=config['grid_layout'])
        else:
            collect={'pulse':collect_pulse,'thermal':collect_thermal,'power':collect_power}[run['mode']]
        data=collect(solver,output,run)
        result=_summary(data,config)
        result['status']='completed'
        dump(output/'summary.json',result)
        manifest['status']='completed'
    except BaseException as exc:
        manifest['status']='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        manifest['error']=dict(type=type(exc).__name__,message=str(exc))
        dump(output/'summary.json',dict(status=manifest['status'],validation_passed=None,error=manifest['error']))
        raise
    finally:
        manifest.update(finished_utc=_now(),elapsed_s=time.perf_counter()-started)
        dump(output/'manifest.json',manifest)
    print(f"  completed; validation_passed={result['validation_passed']}",flush=True)
    return result


def plan_study(path, batch, selected=None):
    path=Path(path).resolve(); valid_name(batch)
    cases=load_study(path)
    if selected:
        selected=set(selected)
        unknown=selected-{c['case']['name'] for c in cases}
        if unknown:
            raise ValueError(f'unknown case selection: {sorted(unknown)}')
        cases=[c for c in cases if c['case']['name'] in selected]
    outputs=[path.parent/batch/'raw'/c['case']['name'] for c in cases]
    for output in outputs:
        if output.exists():
            raise FileExistsError(f'case already exists: {output}')
    return list(zip(cases,outputs))


def run_study(path,batch,*,selected=None,source_cache=None):
    """Run selected cases; source_cache is retained only as an ignored legacy argument."""
    jobs=plan_study(path,batch,selected)
    source=capture_source(ROOT)
    results=[]
    for config,output in jobs:
        result=run_case(config,output,provenance=source)
        results.append(result)
        if synchronized_run(config['run']):
            # Accuracy/stationarity failures remain reviewable results. Physical
            # failures must prevent the next, more expensive experiment.
            physical=result.get('physical_gates',result.get('gates',{}))
            bad=[name for name,passed in physical.items() if not passed and
                 (name.endswith('audit') or name in ('positive_populations','finite_state','impermeable_wall','interface_exchange_recorded'))]
            if bad:
                raise FloatingPointError(f"{config['case']['name']}: physical validation failed: {bad}; later cases not started")
    return results


def analyze_case(path):
    """Read-only reanalysis of completed new or historical open-planar results."""
    path=Path(path)
    config=load_config(path/'effective.yaml')
    previous=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status','completed')!='completed':
        raise ValueError('only completed runs can be analyzed as formal results')
    if 'run' not in config:
        args=manifest['args']
        config['run']=merge(DEFAULTS,dict(mode=args['mode'],refine=args['refine'],length_m=args['length'],
            frequency_Hz=args['frequency'],amplitude_K=args['amplitude'],cycles=args['cycles'],ramp_cycles=args['ramp_cycles']))
        if args['mode']=='thermal':
            config['run']['probes_m']=[p['z_m'] for p in previous['probes']]
    with np.load(path/'trajectory.npz') as data:
        if 'run_completed' in data and not bool(data['run_completed']):
            raise ValueError('trajectory is incomplete')
        result=_summary(data,config,previous)
        if config['run']['mode']=='power' or synchronized_run(config['run']):
            result['status']='completed'
        return result
