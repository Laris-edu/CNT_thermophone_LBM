"""Lim 式 (22) 的 LBM 热流换算对照；保留网格、周期、膜温和守恒诊断。

这里的 p_from_q 是混合 LBM/解析估计，并非 LBM 压力探针的开放域验证。
声频 50--100 kHz、观察距离 0.115 m 对应论文 Fig. 2 的近场条件。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from config_io import load_config
from core.solver import GasSolver2D
from coupling.conjugate import run_levelc_predictor_corrector
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from postproc.multiharmonic_fit import fit_multiharmonic
from reference.lim2013 import (
    LimGas, TABLE1_AIR, characteristic_roots, conductive_heat_flux_hat,
    exact_pressure_hat, film_temperature_hat, near_field_pressure_hat_from_flux,
    near_field_pressure_rms_approximate, rayleigh_distance_m,
)
from simulation.output import OutputLayout, dump
from simulation.provenance import capture_source
from postproc.diagnostics import record, compare, fit_window
from reference.lim2013 import matched_gas

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


class ProgressSolver(GasSolver2D):
    def __init__(self, config, frequency):
        super().__init__(config)
        self.frequency = frequency
        self.last_cycle = 0
        self.started = time.perf_counter()

    def step(self, n_steps=1, boundary_callback=None):
        super().step(n_steps, boundary_callback)
        cycle = int(self.t_lu*self.mapping.lattice.dt_s*self.frequency+1e-9)
        if cycle > self.last_cycle:
            self.last_cycle = cycle
            print(f'  cycle={cycle}, elapsed={time.perf_counter()-self.started:.1f}s', flush=True)


def run_case(args, frequency, refine, output):
    output.mkdir()
    config = load_config(args.config)
    config['numerics'].update(nx=1, ny=args.ny*refine)
    config['lattice']['dx_m'] /= refine
    config['lattice']['dt_s'] /= refine
    solver = ProgressSolver(config, frequency)
    m = solver.mapping
    gas = matched_gas(m)
    dt = m.lattice.dt_s
    steps = int(round(args.cycles/(frequency*dt)))
    config['numerics']['steps'] = steps
    (output/'effective.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    power_density = args.power/args.area
    print(f'[{frequency/1000:g} kHz, refine={refine}] ny={solver.ny}, steps={steps}', flush=True)
    began = time.perf_counter()
    result = run_levelc_predictor_corrector(solver=solver,
        params=FilmOdeParams(C_A_si=args.capacity, T_ref_K=gas.T0_K,
            linear_leak_conductance_si=2*args.beta0, gas_flux_factor=2),
        drive=SinusoidalDrive(0., complex(-power_density), frequency), n_steps=steps)
    arrays = {name: getattr(result, name) for name in (
        't_si', 'T_s_K', 'P_in_si', 'q_g_one_sided_si', 'p_box_mean_Pa',
        'pressure_probe_Pa', 'T_wall_K', 'mass_lu', 'gas_energy_J_m2',
        'wall_energy_increment_J_m2', 'input_work_increment_J_m2', 'leak_work_increment_J_m2')}
    np.savez_compressed(output/'trajectory.npz', **arrays)
    t = result.t_si
    tq = .5*(t[1:]+t[:-1])
    windows = []
    for end in (args.cycles-1, args.cycles):
        hats, quality = {}, {}
        for key, times, values in (
            ('T', t, result.T_s_K), ('q', tq, result.q_g_one_sided_si[1:]),
            ('p_box', t, result.p_box_mean_Pa), ('p_probe', t, result.pressure_probe_Pa),
            ('E_gas', t, result.gas_energy_J_m2)):
            hats[key], quality[key] = fit_window(times, values, frequency, end-1, end)
        windows.append(dict(hats=hats, quality=quality))
    h = windows[-1]['hats']
    stability = {key: abs(h[key]/windows[0]['hats'][key]-1) for key in h}
    T_ref = film_temperature_hat(frequency, -power_density, args.capacity,
        beta0_per_side_W_m2K=args.beta0, gas=gas)
    q_ref = conductive_heat_flux_hat(frequency, T_ref, gas=gas)
    p_from_q = near_field_pressure_hat_from_flux(h['q'], gas)
    # 独立式 (22)；不要用 Lim 的 q 再乘同一个系数充当全部对照。
    p22 = near_field_pressure_rms_approximate(frequency, args.power, args.area, args.capacity,
        beta0_per_side_W_m2K=args.beta0, gas=gas)
    p22_table = near_field_pressure_rms_approximate(frequency, args.power, args.area, args.capacity,
        beta0_per_side_W_m2K=args.beta0, gas=TABLE1_AIR)
    p_at_x = p_from_q*np.exp(-2j*np.pi*frequency*args.distance/gas.c0_m_s)
    p_exact = exact_pressure_hat(frequency, T_ref, args.distance, gas=gas)
    rms = abs(p_from_q)/math.sqrt(2)
    q_cmp = compare(h['q'], q_ref)
    T_cmp = compare(h['T'], T_ref)
    p_cmp = compare(p_at_x, p_exact)
    omega = 2*np.pi*frequency
    harmonic_balance = abs(-power_density-(2*args.beta0+1j*omega*args.capacity)*h['T']-2*h['q'])/power_density
    gas_balance = abs(1j*omega*h['E_gas']/(2*h['q'])-1)
    delta = math.sqrt(2*gas.alpha_m2_s/omega)
    sigma2 = characteristic_roots(frequency, gas)[1]
    gates = dict(
        finite=result.finite,
        energy=max(result.energy_audit.max_relative_residual, result.total_energy_audit.max_relative_residual)<1e-6,
        mass=float(np.max(abs(result.mass_lu/result.mass_lu[0]-1)))<1e-10,
        stationary_T_q=max(stability['T'], stability['q'])<.01,
        pressure_from_q_Eq22=abs(rms/p22-1)<.05,
        film_temperature=abs(T_cmp['amplitude_error_percent'])<5 and abs(T_cmp['phase_error_deg'])<5,
        nearfield_distance=args.distance<=rayleigh_distance_m(frequency, args.area, gas))
    summary = dict(frequency_Hz=frequency, refine=refine, cycles=args.cycles, n_steps=steps,
        elapsed_s=time.perf_counter()-began, gas=asdict(gas), configured_p0_Pa=m.physical.p0_Pa,
        dx_m=m.lattice.dx_m, dt_s=dt, ny=solver.ny, height_m=solver.ny*m.lattice.dx_m,
        delta_T_m=delta, delta_T_over_dx=delta/m.lattice.dx_m,
        half_domain_over_delta_T=.5*solver.ny*m.lattice.dx_m/delta,
        distance_m=args.distance, rayleigh_distance_m=rayleigh_distance_m(frequency, args.area, gas),
        acoustic_attenuation_exponent=float(sigma2.real*args.distance),
        lbm={key: record(z) for key,z in h.items()},
        reference=dict(T_hat=record(T_ref), q_hat=record(q_ref), p_exact_at_x=record(p_exact)),
        p_from_q_rms_Pa=rms, p_Eq22_matched_rms_Pa=p22, p_Eq22_table1_rms_Pa=p22_table,
        p_from_q_at_x=record(p_at_x),
        p_Eq22_error_percent=100*(rms/p22-1), p_Eq22_table1_error_percent=100*(rms/p22_table-1),
        p_exact_comparison=p_cmp, q_comparison=q_cmp, T_comparison=T_cmp,
        box_over_converted_pressure=abs(h['p_box'])/abs(p_from_q),
        window_complex_relative_change=stability,
        windows=[dict(hats={key: record(z) for key,z in w['hats'].items()}, quality=w['quality']) for w in windows],
        film_harmonic_balance_relative=float(harmonic_balance), gas_harmonic_balance_relative=float(gas_balance),
        film_energy_relative=result.energy_audit.max_relative_residual,
        total_energy_relative=result.total_energy_audit.max_relative_residual,
        mass_relative_drift=float(np.max(abs(result.mass_lu/result.mass_lu[0]-1))),
        max_wall_film_difference_K=float(np.max(abs(result.T_wall_K-result.T_s_K))),
        gates=gates)
    dump(output/'summary.json', summary)
    print(f'  p/Eq22 error={summary["p_Eq22_error_percent"]:+.3f}%, '
          f'T phase={T_cmp["phase_error_deg"]:+.3f} deg, window(q)={stability["q"]:.3g}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='实验批次根目录，数据写入 raw/')
    parser.add_argument('--config', type=Path, default=BASE)
    parser.add_argument('--frequencies', type=float, nargs='+', default=[50000., 75000., 100000.])
    parser.add_argument('--refinements', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--ny', type=int, default=64)
    parser.add_argument('--power', type=float, default=.85)
    parser.add_argument('--area', type=float, default=9e-4)
    parser.add_argument('--capacity', type=float, default=1e-3)
    parser.add_argument('--beta0', type=float, default=16.)
    parser.add_argument('--distance', type=float, default=.115)
    args = parser.parse_args()
    positives = [*args.frequencies, args.power, args.area, args.capacity, args.distance]
    if not all(math.isfinite(v) and v>0 for v in positives) or args.cycles<3 or args.ny<24:
        parser.error('频率/功率/面积/热容/距离须正且有限；cycles>=3，ny>=24')
    if any(r<1 for r in args.refinements) or not math.isfinite(args.beta0) or args.beta0<0:
        parser.error('refinements>=1，beta0>=0 且有限')
    if len(set(args.frequencies))!=len(args.frequencies) or len(set(args.refinements))!=len(args.refinements):
        parser.error('扫描点不能重复')
    output = OutputLayout(args.output).create_case()
    (output/'baseline.yaml').write_bytes(args.config.read_bytes())
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        args={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, **capture_source(ROOT),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        scope='AC-only；封闭周期域 LBM 热流与 Lim 式22对照；声压由紧致热层公式换算，非直接压力验证',
        convention='Re[hat*exp(+i*omega*t)]；drive_hat=-P/s；q单侧区间均值；rms=abs(hat)/sqrt(2)',
        thresholds=dict(amplitude_percent=5, phase_deg=5, stationary_complex_relative=.01,
            energy_relative=1e-6, mass_relative=1e-10),
        completed=False, cases=[])
    dump(output/'summary.json', manifest)
    for frequency in args.frequencies:
        for refine in args.refinements:
            folder = output/f'f{frequency:g}_r{refine}'
            try:
                result = run_case(args, frequency, refine, folder)
            except Exception as exc:
                manifest['failed_case'] = dict(frequency_Hz=frequency, refine=refine, error=repr(exc))
                dump(output/'summary.json', manifest)
                raise
            manifest['cases'].append(result)
            dump(output/'summary.json', manifest)
    manifest['completed'] = True
    dump(output/'summary.json', manifest)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
