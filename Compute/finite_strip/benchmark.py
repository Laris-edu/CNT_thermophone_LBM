"""Time a few real coupled power steps; this is not a completed simulation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from Compute.finite_strip.run import load_cases
from core.finite_strip_solver import FiniteStripSolver
from coupling.film_ode import FilmOdeParams
from coupling.finite_strip import audit_strip_film_step, solve_strip_film_midpoint
from simulation.provenance import capture_cupy_runtime, capture_numba_runtime, capture_source
from simulation.study import ROOT


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).with_name('study_30mm.yaml'))
    parser.add_argument('--case',default='power_r2')
    parser.add_argument('--backend',choices=('numba','cupy'),default='cupy')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.warmup<0 or args.steps<1:
        parser.error('warmup must be nonnegative and steps must be positive')
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    _,cases=load_cases(args.study.resolve())
    config=deepcopy(cases[args.case]); run=config['strip']
    if run['kind']!='power':
        raise ValueError('benchmark requires a power case with actual film coupling')
    config['numerics']['collision_backend']=run['collision_backend']=args.backend
    cp=None
    if args.backend=='cupy':
        import cupy as cp
    def synchronize():
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
    def runtime(cells):
        value=(capture_cupy_runtime if cp is not None else capture_numba_runtime)(cells)
        if cp is not None:
            pool=cp.get_default_memory_pool()
            value.update(memory_pool_used_bytes=int(pool.used_bytes()),
                         memory_pool_total_bytes=int(pool.total_bytes()))
        return value
    record=dict(kind='coupled_step_benchmark',status='benchmark_running',
        created_utc=datetime.now(timezone.utc).isoformat(),study=str(args.study.resolve()),
        case=args.case,collision_backend=args.backend,effective=config,
        nx=config['numerics']['nx'],ny=config['numerics']['ny'],
        configured_steps=run['steps'],dt_s=config['lattice']['dt_s'],
        warmup_steps=args.warmup,timed_steps=args.steps,
        estimated_max_run_seconds_scope='Step-only estimate; excludes sampling, output, '
            'stationarity checks and any later-state change in coupling cost.',
        scope='Short benchmark only; no trajectory completion or validation claim.',
        benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **capture_source(ROOT))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        try:
            cold_start=time.perf_counter()
            solver=FiniteStripSolver(config,film_width_m=run['film_width_m'],
                sponge_cells=run['sponge_cells'],sponge_rate=run['sponge_rate'],
                pml_cells=run.get('pml_cells',0),pml_target_return=run['pml_target_return'],
                pml_alpha_Hz=run['pml_alpha_Hz'])
            solver._require_state()
            dt=solver.mapping.lattice.dt_s
            temperature=np.full(len(solver.film_x_m),run['initial_temperature_K'])
            params=FilmOdeParams(C_A_si=run['C_A_J_m2K'],T_ref_K=run['initial_temperature_K'],
                linear_leak_conductance_si=run['G_W_m2K'],gas_flux_factor=2.)
            maxima=dict(gas_energy_residual_J_m=0.,film_energy_residual_J_m=0.,
                       total_energy_residual_J_m=0.,gas_mass_residual_lu=0.)
            def step():
                nonlocal temperature
                power=run['mean_power_W_m2']*(1-np.cos(2*np.pi*run['frequency_Hz']*(solver.t_lu+.5)*dt))
                trial=[]
                def solve(exchange):
                    trial.append(solve_strip_film_midpoint(params=params,T_s_K=temperature,
                        dx_m=solver.film_dx_m,sheet_conductance_W_K=run['sheet_conductance_W_K'],
                        dt_s=dt,power_W_m2=power,exchange_J_m2=exchange,exchange_is_local=True))
                    return trial[-1]
                audit=solver.step_wall_energy(solve)
                new=2*trial[-1]-temperature
                film=audit_strip_film_step(params=params,T_old_K=temperature,T_new_K=new,
                    dx_m=solver.film_dx_m,dt_s=dt,power_W_m2=power,
                    wall_energy_J_m2=audit.wall_energy_J_m2)
                total=film.storage_change_J_m+2*(audit.gas_energy_after_J_m-audit.gas_energy_before_J_m)
                total+=film.leak_energy_J_m-film.input_energy_J_m-2*(audit.outer_energy_J_m
                    +audit.sponge_energy_J_m+audit.pml_energy_J_m)
                for name,value in zip(maxima,(audit.energy_residual_J_m,film.residual_J_m,
                                             total,audit.mass_residual_lu)):
                    maxima[name]=max(maxima[name],abs(float(value)))
                temperature=new
            synchronize()
            record['setup_elapsed_s']=time.perf_counter()-cold_start
            record['runtime_started']=runtime(solver.nx*solver.ny)
            warmup_start=time.perf_counter()
            for _ in range(args.warmup):
                step()
            synchronize()
            record['warmup_elapsed_s']=time.perf_counter()-warmup_start
            record['cold_elapsed_s']=time.perf_counter()-cold_start
            started=time.perf_counter()
            for _ in range(args.steps):
                step()
            synchronize()
            elapsed=time.perf_counter()-started
            record.update(status='benchmark_completed',timed_elapsed_s=elapsed,
                seconds_per_step=elapsed/args.steps,
                estimated_max_run_seconds=run['steps']*elapsed/args.steps,
                accepted_steps=int(solver.t_lu),physical_time_s=solver.t_lu*dt,
                max_independent_step_residuals=maxima,runtime=runtime(solver.nx*solver.ny))
        except BaseException as exc:
            record.update(status='benchmark_failed',error=dict(type=type(exc).__name__,message=str(exc)))
            try:
                record['runtime']=runtime(record['nx']*record['ny'])
            except Exception as runtime_error:
                record['runtime_error']=str(runtime_error)
            raise
        finally:
            record['finished_utc']=datetime.now(timezone.utc).isoformat()
            json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps({key:record[key] for key in ('status','nx','ny','seconds_per_step',
        'estimated_max_run_seconds','max_independent_step_residuals')},indent=2),flush=True)


if __name__=='__main__':
    main()
