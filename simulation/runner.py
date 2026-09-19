"""Four-file case lifecycle and sequential study runner."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from config_io import load_config
from core.planar_open_solver import PlanarOpenSolver
from core.unit_mapping import create_unit_mapping
from postproc.planar_open import analyze_pulse, analyze_thermal
from simulation.output import dump
from simulation.planar import collect_pulse, collect_thermal
from simulation.provenance import capture_source
from simulation.study import ROOT, DEFAULTS, load_study, merge, valid_name, validate_effective


def _now():
    return datetime.now(timezone.utc).isoformat()


def source_reference(snapshot, output):
    """Use a portable relative reference, or an absolute path across Windows drives."""
    try:
        return Path(os.path.relpath(snapshot,output)).as_posix()
    except ValueError:
        return Path(snapshot).resolve().as_posix()


def _summary(data, config, previous=None):
    mapping=create_unit_mapping(config)
    run=config['run']
    result=(analyze_pulse if run['mode']=='pulse' else analyze_thermal)(data,mapping,run,previous)
    result.update(mapping=mapping.to_metadata(),steps=len(data['t_s']),
        actual_length_m=config['numerics']['ny']*mapping.lattice.dx_m,
        validation_passed=bool(result['pass_reflection'] if run['mode']=='pulse' else all(result['gates'].values())))
    return result


def run_case(effective, output, *, source_cache=None, provenance=None):
    """Run a fully resolved configuration into a NEW directory. Never overwrites data."""
    config=deepcopy(effective)
    validate_effective(config)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'case already exists: {output}')
    # Provenance is prepared before creating the case directory; it is shared per batch.
    source=deepcopy(provenance if provenance is not None else
        capture_source(ROOT,source_cache or ROOT/'Compute/_sources'))
    for name,digest in source['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('source changed during the batch; use a new process/batch')
    snapshot=source.pop('snapshot')
    if snapshot:
        snapshot=source_reference(snapshot,output)
    run=config['run']; started=time.perf_counter()
    manifest=dict(created_utc=_now(),finished_utc=None,status='running',elapsed_s=None,
        python=platform.python_version(),numpy=np.__version__,pyyaml=yaml.__version__,
        source_snapshot=snapshot,**source,
        args=dict(mode=run['mode'],refine=run['refine'],length=run['length_m'],frequency=run['frequency_Hz'],
            amplitude=run['amplitude_K'],cycles=run['cycles'],ramp_cycles=run['ramp_cycles'],output=str(output)),
        pressure='direct rho*theta; no heat-flux conversion',
        geometry='nx=1 planar upper half-space; z=(row+0.5)dx; open top; diffuse thermal bottom',
        audit_columns=['wall_mass_lu','top_mass_lu','wall_energy_J_m2','top_energy_J_m2',
            'mass_residual_lu','energy_residual_J_m2'])
    output.mkdir(parents=True,exist_ok=False)
    try:
        (output/'effective.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
        manifest['effective_sha256']=hashlib.sha256((output/'effective.yaml').read_bytes()).hexdigest()
        dump(output/'manifest.json',manifest)
        np.savez_compressed(output/'trajectory.npz',t_s=np.array([]),run_completed=False)
        dump(output/'summary.json',dict(status='running',validation_passed=None))
        solver=PlanarOpenSolver(config)
        print(f"{config['case']['name']}: {run['mode']}, ny={solver.ny}, dx={solver.mapping.lattice.dx_m:g}",flush=True)
        collect=collect_pulse if run['mode']=='pulse' else collect_thermal
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
    jobs=plan_study(path,batch,selected)
    source=capture_source(ROOT,source_cache or ROOT/'Compute/_sources')
    return [run_case(config,output,provenance=source) for config,output in jobs]


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
        return _summary(data,config,previous)
