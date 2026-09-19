"""Resolve editable study descriptions into complete, independent case configs."""
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import re

import numpy as np
import yaml

from config_io import load_config
from core.unit_mapping import create_unit_mapping

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'
DEFAULTS = dict(mode='thermal', refine=4, length_m=1e-3, frequency_Hz=50000.,
    amplitude_K=.1, cycles=6., ramp_cycles=2., probes_m=[.12e-3, .30e-3, .60e-3],
    sampling=dict(profile_cycles=2., profile_samples_per_cycle=128),
    analysis=dict(harmonics=3, detrend_order=1, window_cycles=1., amplitude_percent=5.,
        phase_deg=5., stationary_relative=.01, reflection_ratio=.01,
        wall_mass_lu=1e-12, mass_residual_lu=1e-10, energy_residual_J_m2=1e-10,
        cumulative_mass_lu=1e-8, cumulative_energy_J_m2=1e-8),
    pulse=dict(epsilon=1e-5, center_fraction=.3, width_fraction=.045,
        probe_fraction=.65, duration_crossings=1.65, window_fraction=.16))


def valid_name(value):
    """Portable directory component, including Windows reserved-name protection."""
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)
        or value.endswith('.') or value.split('.')[0].upper() in
            {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}):
        raise ValueError(f'invalid directory name: {value!r}')
    return value


def merge(base, update):
    result = deepcopy(base)
    if not isinstance(update, dict):
        raise ValueError('parameters must be a mapping')
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _finite(value, name, *, minimum=0., strict=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        or (value <= minimum if strict else value < minimum)):
        raise ValueError(f'{name} must be finite and {">" if strict else ">="} {minimum}')


def validate_run(run):
    if set(run) != set(DEFAULTS):
        raise ValueError(f'unknown/missing run parameters: {set(run) ^ set(DEFAULTS)}')
    if run['mode'] not in ('thermal','pulse'):
        raise ValueError('mode must be thermal or pulse')
    for key in ('sampling','analysis','pulse'):
        if not isinstance(run[key],dict) or set(run[key]) != set(DEFAULTS[key]):
            raise ValueError(f'unknown/missing {key} parameters')
    for key in ('length_m','frequency_Hz','amplitude_K','cycles'):
        _finite(run[key],key)
    if run['amplitude_K'] >= 1:
        raise ValueError('current small-signal thermal cases require amplitude_K < 1')
    _finite(run['ramp_cycles'],'ramp_cycles',strict=False)
    for value, name in ((run['refine'],'refine'),
            (run['sampling']['profile_samples_per_cycle'],'profile_samples_per_cycle'),
            (run['analysis']['harmonics'],'harmonics')):
        if isinstance(value,bool) or not isinstance(value,int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    order=run['analysis']['detrend_order']
    if isinstance(order,bool) or not isinstance(order,int) or order not in (0,1):
        raise ValueError('detrend_order must be 0 or 1')
    for key,value in run['analysis'].items():
        if key not in ('harmonics','detrend_order'):
            _finite(value,key)
    _finite(run['sampling']['profile_cycles'],'profile_cycles')
    if run['mode']=='thermal':
        if run['cycles'] < run['ramp_cycles'] + 2*run['analysis']['window_cycles']:
            raise ValueError('need two analysis windows after the temperature ramp')
        if not isinstance(run['probes_m'],list) or not run['probes_m']:
            raise ValueError('probes_m must be a nonempty list')
        for value in run['probes_m']:
            _finite(value,'probe position')
    pulse=run['pulse']
    for key,value in pulse.items():
        _finite(value,key)
    if not (pulse['epsilon'] < .01 and pulse['center_fraction'] < pulse['probe_fraction'] < 1
            and pulse['width_fraction'] < 1 and pulse['window_fraction'] < 1):
        raise ValueError('invalid small-amplitude pulse geometry')


def validate_effective(config):
    run=config['run']
    validate_run(run)
    valid_name(config['case']['name'])
    mapping=create_unit_mapping(config)
    nx,ny=config['numerics']['nx'],config['numerics']['ny']
    if nx != 1 or not isinstance(ny,int) or ny < 8 or config['numerics']['boundary'] != 'planar_open_y':
        raise ValueError('open planar cases require nx=1, ny>=8, planar_open_y')
    z=(np.arange(ny)+.5)*mapping.lattice.dx_m
    if run['mode']=='thermal':
        probes=run['probes_m']
        if min(probes)<z[0] or max(probes)>=z[-1]:
            raise ValueError('domain must contain all requested probes')
        if len({int(np.argmin(abs(z-p))) for p in probes}) != len(probes):
            raise ValueError('probes must resolve to different grid cells')
        samples=run['analysis']['window_cycles'] / (run['frequency_Hz']*mapping.lattice.dt_s)
        if samples < 2*run['analysis']['harmonics']+run['analysis']['detrend_order']+3:
            raise ValueError('analysis window has too few samples')
        steps=int(np.ceil(run['cycles']/run['frequency_Hz']/mapping.lattice.dt_s))
    else:
        p=run['pulse']; H=ny*mapping.lattice.dx_m
        probe=z[np.argmin(abs(z-p['probe_fraction']*H))]/H
        if (probe-p['center_fraction'] <= p['window_fraction'] or
            1-probe <= p['window_fraction'] or
            p['duration_crossings'] < 2-probe-p['center_fraction']+p['window_fraction']):
            raise ValueError('pulse windows must be separated and covered by duration')
        steps=int(np.ceil(p['duration_crossings']*H/mapping.physical.c0_m_s/mapping.lattice.dt_s))
        crossing=H/mapping.physical.c0_m_s
        half=p['window_fraction']*crossing
        dt=mapping.lattice.dt_s
        for center in ((probe-p['center_fraction'])*crossing,
                       (2-probe-p['center_fraction'])*crossing):
            first=max(1,math.floor((center-half)/dt)+1)
            last=min(steps,math.ceil((center+half)/dt)-1)
            if first > last:
                raise ValueError('pulse analysis windows contain no discrete samples')
    if config['numerics']['steps'] != steps:
        raise ValueError('effective steps do not match the requested run')
    return config


def prepare_case(parameters, base_config=BASE):
    parameters=deepcopy(parameters)
    name=valid_name(parameters.pop('name','case'))
    run=merge(DEFAULTS,parameters)
    validate_run(run)
    config=deepcopy(base_config) if isinstance(base_config,dict) else load_config(Path(base_config))
    config['lattice']['dx_m'] /= run['refine']
    config['lattice']['dt_s'] /= run['refine']
    ny=int(round(run['length_m']/config['lattice']['dx_m']))
    config['numerics'].update(nx=1,ny=ny,boundary='planar_open_y')
    config['case'].update(name=name,purpose='single-sided open planar '+run['mode'])
    config['run']=run
    mapping=create_unit_mapping(config)
    seconds=(run['cycles']/run['frequency_Hz'] if run['mode']=='thermal' else
        run['pulse']['duration_crossings']*ny*mapping.lattice.dx_m/mapping.physical.c0_m_s)
    config['numerics']['steps']=int(np.ceil(seconds/mapping.lattice.dt_s))
    return validate_effective(config)


def load_study(path):
    path=Path(path)
    study=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(study,dict) or set(study)-{'base_config','defaults','cases','sweeps'}:
        raise ValueError('study accepts base_config, defaults, cases and sweeps')
    base=(path.parent/Path(study.get('base_config',BASE))).resolve()
    defaults=study.get('defaults',{})
    specs=[]
    explicit=study.get('cases',[]); sweeps=study.get('sweeps',[])
    if not isinstance(explicit,list) or not isinstance(sweeps,list):
        raise ValueError('cases and sweeps must be lists')
    for case in explicit:
        if not isinstance(case,dict) or 'name' not in case:
            raise ValueError('each case needs a name')
        specs.append(merge(defaults,case))
    for sweep in sweeps:
        if not isinstance(sweep,dict) or set(sweep)-{'name','parameters','overrides'}:
            raise ValueError('sweep accepts name, parameters and overrides')
        name=valid_name(sweep['name']); axes=sweep['parameters']
        if (not isinstance(axes,dict) or not axes or set(axes)-set(DEFAULTS) or
            any(not isinstance(v,list) or not v for v in axes.values())):
            raise ValueError('sweep parameters must be nonempty lists of run parameters')
        for values in product(*axes.values()):
            def label(value):
                if isinstance(value,(str,int,float)) and not isinstance(value,bool):
                    return str(value) if isinstance(value,str) else f'{value:g}'
                raise ValueError('sweep axis values must be scalars')
            generated=name+''.join(f'__{k}_{label(v)}' for k,v in zip(axes,values))
            spec=merge(defaults,sweep.get('overrides',{}))
            spec.update(dict(zip(axes,values)),name=generated)
            specs.append(spec)
    names=[valid_name(s['name']).casefold() for s in specs]
    if len(names) != len(set(names)):
        raise ValueError('duplicate case name (case insensitive)')
    if not specs:
        raise ValueError('study contains no cases')
    return [prepare_case(spec,base) for spec in specs]
