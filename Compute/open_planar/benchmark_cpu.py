"""Short CPU benchmark, not a physical case or restart; writes one diagnostic NPZ."""
import argparse
from copy import deepcopy
from dataclasses import astuple
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from core.planar_open_solver import PlanarOpenSolver
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from coupling.planar_open import step_open_film
from simulation.provenance import capture_source
from simulation.study import ROOT, load_study


def measure(config, warm, steps):
    solver = PlanarOpenSolver(config)
    mapping = solver.mapping
    film = config['run']['film']
    params = FilmOdeParams(C_A_si=film['C_A_J_m2K'], T_ref_K=film['T_ref_K'],
        linear_leak_conductance_si=film['G_W_m2K'], gas_flux_factor=2.)
    mean = config['run']['power']['mean_W'] / config['run']['power']['area_m2']
    frequency = config['run']['frequency_Hz']
    drive = SinusoidalDrive(mean, -mean, frequency)
    rows = [int(np.argmin(abs(solver.z_m-z))) for z in config['run']['probes_m']]
    temperature = 302.5 if warm else 300.
    if warm:
        # Synthetic hot initial state to exercise nonzero wall exchange and flow.
        gas_T = 300 + 2.5 * np.exp(-solver.z_m[:, None] / 50e-6)
        velocity = np.zeros((solver.ny, 1, 2))
        velocity[..., 1] = .001 / mapping.velocity_scale
        solver.initialize_from_macro(300 / gas_T, velocity, gas_T / mapping.temperature_scale)
        solver.t_lu = int(round(.25 / frequency / mapping.lattice.dt_s))
    solver.get_macro()
    history = []
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for _ in range(steps):
        step = step_open_film(solver=solver, params=params, T_s_K=temperature, drive=drive)
        temperature = step.T_s_K
        state = solver.get_macro()
        history.append([temperature, *astuple(step.audit),
            *state.p[rows, 0] * mapping.pressure_scale,
            float(np.min(solver.f)), float(np.min(solver.g))])
    elapsed = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return dict(wall_s=elapsed, cpu_s=cpu, ms_per_step=1000*elapsed/steps), np.asarray(history), solver.f, solver.g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--backend', choices=('numpy', 'numba'), default='numpy')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps <= 0 or args.repeats <= 0:
        raise ValueError('steps and repeats must be positive')
    config = load_study(Path(__file__).with_name('e1_study.yaml'))[0]
    config['numerics']['collision_backend'] = args.backend
    warmup_start = time.perf_counter()
    measure(config, False, 16)  # Warm up fixed projectors and imports.
    warmup_s = time.perf_counter() - warmup_start
    source = capture_source(ROOT)
    metadata = dict(steps=args.steps, repeats=args.repeats,
        git_commit=source['git_commit'], git_dirty=source['git_dirty'],
        source_clean=source['source_clean'], source_sha256=source['source_sha256'],
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, config=config, timings={},
        collision_backend=args.backend, warmup_s=warmup_s)
    if args.backend == 'numba':
        metadata.update(numba=version('numba'), llvmlite=version('llvmlite'))
    arrays = {}
    for name, warm in [('cold', False), ('warm', True)]:
        timings = []
        for _ in range(args.repeats):
            timing, history, f, g = measure(config, warm, args.steps)
            timings.append(timing)
        metadata['timings'][name] = timings
        arrays.update({name+'_history': history, name+'_f': f, name+'_g': g})
        print(name, json.dumps(timings), flush=True)
    if args.compare:
        with np.load(args.compare) as previous:
            old = json.loads(str(previous['metadata']))
            if old['steps'] != args.steps:
                raise ValueError('comparison requires identical step counts')
            reference_config = deepcopy(old.get('config', config))
            compared_config = deepcopy(config)
            for compared in (reference_config, compared_config):
                compared['numerics'].pop('collision_backend', None)
            if reference_config != compared_config:
                raise ValueError('comparison requires the same effective configuration')
            metadata['comparison'] = dict(reference=str(args.compare), max_abs={}, speedup={})
            for name, value in arrays.items():
                metadata['comparison']['max_abs'][name] = float(np.max(abs(value-previous[name])))
            for name in ('cold', 'warm'):
                metadata['comparison']['speedup'][name] = float(
                    np.median([r['wall_s'] for r in old['timings'][name]]) /
                    np.median([r['wall_s'] for r in metadata['timings'][name]]))
                actual, expected = arrays[name+'_history'], previous[name+'_history']
                np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=1e-9)
                np.testing.assert_allclose(actual[:, 1:7], expected[:, 1:7], rtol=0, atol=1e-11)
                np.testing.assert_allclose(actual[:, 7:-2], expected[:, 7:-2], rtol=0, atol=1e-7)
                for suffix in ('f', 'g'):
                    np.testing.assert_allclose(arrays[name+'_'+suffix], previous[name+'_'+suffix], rtol=0, atol=2e-14)
            metadata['comparison']['numerical_equivalence_passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, metadata=json.dumps(metadata), **arrays)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('source_sha256', 'config')}, indent=2))


if __name__ == '__main__':
    main()
