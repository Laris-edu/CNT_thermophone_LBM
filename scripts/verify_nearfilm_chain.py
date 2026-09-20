"""现有一维近膜链路诊断；只增加观测，不修改求解器、热壁或反馈。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.integrate import solve_bvp
import yaml

from boundary.wall_thermal_grad import make_bottom_grad_wall_callback
from boundary.wall_thermal_mass_neutral import make_symmetric_mass_neutral_wall_callback
from config_io import load_config
from core.solver import GasSolver2D
from coupling.conjugate import extract_bottom_wall_heat_flux_si, run_levelc_predictor_corrector
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from postproc.multiharmonic_fit import fit_multiharmonic
from reference.constants import PhysicalParams
from reference.continuum_1d_freq import solve_level_C_frequency
from simulation.output import OutputLayout
from simulation.provenance import capture_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


def complex_record(z):
    z = complex(z)
    return dict(re=z.real, im=z.imag, abs=abs(z), phase_deg=float(np.angle(z, deg=True)))


def closed_nsf_reference(mapping, frequency, ny, probe_rows):
    """独立线性 D2 NSF 边值问题：u(0)=u(L)=0, T(0)=1 K, q(L)=0。

    状态 [u,T,p,q]，线性化于实际映射的 EOS；D2 零体黏度 mu_L=mu。
    坐标缩放为 y/L。全周期域关于热带和中点对称，L=ny*dx/2。
    """
    ph = mapping.physical
    rho, T0, cp, k = ph.rho0_kg_m3, ph.T0_K, ph.cp_J_kgK, ph.kg_W_mK
    p0 = mapping.lattice.rho_ref_lu * mapping.theta_ref_lu * mapping.pressure_scale
    omega = 2 * np.pi * frequency
    muL = rho * ph.nu0_m2_s
    length = ny * mapping.lattice.dx_m / 2
    matrix = np.zeros((4, 4), complex)
    matrix[0, 1], matrix[0, 2] = 1j * omega / T0, -1j * omega / p0
    matrix[1, 3] = -1 / k
    visc = 1 + 1j * omega * muL / p0
    matrix[2, 0] = -1j * omega * rho / visc
    matrix[2, 3] = -1j * omega * muL / (k * T0 * visc)
    matrix[3, 1], matrix[3, 2] = -1j * omega * rho * cp, 1j * omega
    grid = np.linspace(0, 1, 201)
    guess = np.zeros((4, grid.size), complex)
    guess[1] = 1
    def bc(left, right):
        return np.array([left[0], left[1] - 1, right[0], right[3]])
    sol = solve_bvp(lambda x, s: length * matrix @ s, bc, grid, guess,
                    tol=1e-8, max_nodes=10000)
    if not sol.success:
        raise RuntimeError(sol.message)
    ys = np.asarray(probe_rows) * mapping.lattice.dx_m
    out = sol.sol(ys / length)
    dense_y = np.linspace(0, length, 2001)
    dense = sol.sol(dense_y / length)
    ehat = np.trapezoid(dense[2] / (ph.gamma - 1), dense_y)
    residual = abs(1j * omega * ehat / sol.sol(0)[3] - 1)
    return dict(q_wall=complex(sol.sol(0)[3]), q_row1=complex(sol.sol(2 / ny)[3]),
                p=out[2], T=out[1], energy_identity_relative=float(residual),
                bvp_max_rms_residual=float(np.max(sol.rms_residuals)))


class ObservedSolver(GasSolver2D):
    """在原 step 之后读取场；步数及温升上限仅用于终止诊断。"""
    def __init__(self, config, stride, frequency):
        super().__init__(config)
        self.stride, self.frequency = stride, frequency
        self.samples = []
        self.started = time.perf_counter()
        self.probe_rows = [1, 5, 10, self.ny // 2]

    def observe(self):
        m = self.get_macro()
        temp = m.theta * self.mapping.temperature_scale
        if not np.isfinite(temp).all() or np.max(abs(temp - self.mapping.physical.T0_K)) > 5:
            raise RuntimeError('诊断终止：非有限场或气体温偏超过5 K；不作裁剪')
        scale = self.mapping.pressure_scale * self.mapping.lattice.dx_m
        self.samples.append(dict(
            t=self.t_lu * self.mapping.lattice.dt_s,
            energy=float(np.sum(m.E_tot) / self.nx * scale),
            energy_nonwall=float(np.sum(m.E_tot[1:]) / self.nx * scale),
            mass=float(np.sum(m.rho)),
            T_wall=float(np.mean(temp[0])),
            q_row1=extract_bottom_wall_heat_flux_si(self),
            pressure=np.mean(m.p[self.probe_rows], axis=1) * self.mapping.pressure_scale,
            temperature=np.mean(temp[self.probe_rows], axis=1),
            p_mean=float(np.mean(m.p) * self.mapping.pressure_scale),
            x_spread=float(np.max(np.ptp(temp, axis=1))),
        ))

    def step(self, n_steps=1, boundary_callback=None):
        super().step(n_steps, boundary_callback)
        if self.t_lu % self.stride == 0:
            self.observe()
        if self.t_lu % 10000 == 0:
            print(f'步数={self.t_lu}, 周期={self.t_lu*self.mapping.lattice.dt_s*self.frequency:.3f}, '
                  f'耗时={time.perf_counter()-self.started:.1f}s', flush=True)


def fit(t, x, frequency):
    return fit_multiharmonic(t, x, 2*np.pi*frequency, n_harmonics=3, detrend_order=1)


def analyze(solver, args, coupled, output):
    samples = solver.samples
    data = {k: np.asarray([s[k] for s in samples]) for k in samples[0]}
    t = data['t']
    out = dict(mass_relative_drift=float(np.max(abs(data['mass']/data['mass'][0]-1))),
               max_x_temperature_spread_K=float(np.max(data['x_spread'])), windows={})
    for name, start, end in [('previous', args.cycles-2, args.cycles-1),
                             ('last', args.cycles-1, args.cycles)]:
        mask = (t*args.frequency >= start-1e-9) & (t*args.frequency < end-1e-9)
        hats = {}
        quality = {}
        for key in ['T_wall', 'q_row1', 'energy', 'energy_nonwall', 'p_mean']:
            f = fit(t[mask], data[key][mask], args.frequency)
            hats[key] = f.harmonic(1)
            quality[key] = dict(residual_rms=f.residual_rms,
                                h2_relative=f.amplitude(2)/max(f.amplitude(1), 1e-30))
        for i, row in enumerate(solver.probe_rows):
            hats[f'p_row{row}'] = fit(t[mask], data['pressure'][mask,i], args.frequency).harmonic(1)
        if coupled is not None:
            tc = coupled.t_si
            mc = (tc*args.frequency >= start-1e-9) & (tc*args.frequency < end-1e-9)
            for key, array in [('T_film', coupled.T_s_K), ('P', coupled.P_in_si),
                               ('q_feedback', coupled.q_g_one_sided_si)]:
                if key == 'q_feedback' and coupled.q_sample_kind == 'preceding_interval_mean':
                    # 区间平均通量必须在区间中点拟合，并去掉首个占位值。
                    tq = .5*(tc[1:]+tc[:-1])
                    mq = (tq*args.frequency >= start-1e-9) & (tq*args.frequency < end-1e-9)
                    hats[key] = fit(tq[mq], array[1:][mq], args.frequency).harmonic(1)
                else:
                    hats[key] = fit(tc[mc], array[mc], args.frequency).harmonic(1)
            out['film_integral_audit_relative'] = coupled.energy_audit.max_relative_residual
            out['q_sample_kind'] = coupled.q_sample_kind
            if coupled.total_energy_audit is not None:
                out['total_integral_audit_relative'] = coupled.total_energy_audit.max_relative_residual
            out['max_predictor_corrector_delta_K'] = float(np.max(abs(coupled.predictor_corrector_delta_K)))
            out['max_wall_vs_corrected_film_K'] = float(np.max(abs(coupled.wall_temperature_error_K)))
        iw = 2j*np.pi*args.frequency
        hats['q_energy_all'] = iw*hats['energy']/2
        hats['q_energy_nonwall'] = iw*hats['energy_nonwall']/2
        Y = hats['q_row1']/hats['T_wall']
        window = dict(hats={k:complex_record(v) for k,v in hats.items()}, fit_quality=quality,
                      Y_row1=complex_record(Y),
                      q_energy_over_row1=complex_record(hats['q_energy_all']/hats['q_row1']),
                      q_nonwall_energy_over_row1=complex_record(hats['q_energy_nonwall']/hats['q_row1']))
        if coupled is not None:
            P, Tf, qf = hats['P'], hats['T_film'], hats['q_feedback']
            window['film_frequency_residual_relative'] = abs(P-2*qf-iw*args.capacity*Tf)/abs(P)
            window['film_plus_gas_frequency_residual_relative'] = abs(P-iw*(args.capacity*Tf+hats['energy']))/abs(P)
        out['windows'][name] = window
    prev, last = [out['windows'][n]['hats'] for n in ['previous','last']]
    tracked = ['T_wall','q_row1','p_mean'] + [k for k in last if k.startswith('p_row')]
    tracked += [k for k in ['T_film','q_feedback'] if k in last]
    out['window_relative_change'] = {k: abs(complex(last[k]['re'],last[k]['im'])-
         complex(prev[k]['re'],prev[k]['im']))/max(abs(complex(prev[k]['re'],prev[k]['im'])),1e-30)
         for k in tracked}
    ref = closed_nsf_reference(solver.mapping,args.frequency,solver.ny,solver.probe_rows)
    out['closed_D2_NSF_per_K'] = {k:complex_record(ref[k]) for k in ['q_wall','q_row1']}
    out['closed_D2_NSF_per_K'].update(
        p=[complex_record(z) for z in ref['p']],
        energy_identity_relative=ref['energy_identity_relative'],
        bvp_max_rms_residual=ref['bvp_max_rms_residual'])
    np.savez_compressed(output/'observations.npz',**data)
    if coupled is not None:
        np.savez_compressed(output/'coupled.npz', **{k:getattr(coupled,k) for k in
            ['t_si','T_s_K','P_in_si','q_g_one_sided_si','pressure_probe_Pa',
             'T_wall_K','mass_lu','p_box_mean_Pa','predictor_corrector_delta_K',
             'gas_energy_J_m2','wall_energy_increment_J_m2','input_work_increment_J_m2',
             'leak_work_increment_J_m2'] if getattr(coupled,k) is not None})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['A','B','C'],required=True)
    parser.add_argument('--scheme',choices=['conservative','heun_picard1'],default='conservative')
    parser.add_argument('--wall',choices=['mass_neutral','thermal_grad'],default='mass_neutral')
    parser.add_argument('--ny',type=int,default=64)
    parser.add_argument('--nx',type=int,default=1)
    parser.add_argument('--refine',type=int,default=1,help='dx/dt同比缩小，ny同比增加，保持物理域长')
    parser.add_argument('--cycles',type=int,default=3)
    parser.add_argument('--frequency',type=float,default=10000)
    parser.add_argument('--amplitude',type=float,default=0.01,help='A: K; B: 单侧热流W/m²; C: 面功率W/m²')
    parser.add_argument('--capacity',type=float,default=7e-4,help='明确声明的验证夹具，不是器件实参')
    parser.add_argument('--relax',type=float,default=1.0)
    parser.add_argument('--stride',type=int,default=16)
    args = parser.parse_args()
    if args.cycles < 3 or args.ny < 24 or args.nx < 1 or args.frequency <= 0 or args.stride < 1 or args.refine<1:
        parser.error('要求 cycles>=3, ny>=24, nx>=1, frequency>0, stride>=1, refine>=1')
    if args.mode == 'B' and args.wall != 'mass_neutral':
        parser.error('B only supports the symmetric mass-neutral wall')
    output = OutputLayout(args.output).create_case()
    config = load_config(args.config)
    config['numerics'].update(nx=args.nx,ny=args.ny*args.refine)
    config['lattice']['dx_m'] /= args.refine
    config['lattice']['dt_s'] /= args.refine
    config['numerics']['steps'] = int(round(args.cycles/(args.frequency*config['lattice']['dt_s'])))
    (output/'baseline.yaml').write_bytes(args.config.read_bytes())
    (output/'effective.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    solver = ObservedSolver(config,args.stride,args.frequency)
    solver.probe_rows = [args.refine*j for j in [1,5,10,args.ny//2]]
    ph, mp = solver.mapping.physical, solver.mapping
    actual_p0 = mp.lattice.rho_ref_lu*mp.theta_ref_lu*mp.pressure_scale
    matched = PhysicalParams(T0=ph.T0_K,p0=actual_p0,rho0=ph.rho0_kg_m3,c0=ph.c0_m_s,
        gamma=ph.gamma,cp=ph.cp_J_kgK,kg=ph.kg_W_mK,nu0=ph.nu0_m2_s,C_A=args.capacity,beta0=0)
    reference = solve_level_C_frequency(args.frequency,1+0j,params=matched)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),interpreter=sys.executable,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        configured_p0_Pa=ph.p0_Pa, actual_EOS_p0_Pa=actual_p0,
        alpha_m2_s=matched.alpha0,delta_T_m=reference.delta_T,
        halfspace_existing_reference_per_W_m2=dict(T=complex_record(reference.T_s_hat),
            q=complex_record(reference.q_g_hat),
            p=[complex_record(z) for z in reference.p_hat_probes],
            probe_y_m=reference.probes_y.tolist()),
        scope='小扰动/有限周期诊断；周期回绕域，不是开放半空间或器件认证',
        thresholds=dict(window_complex_relative=0.01,film_energy_relative=0.01,
                        interface_energy_relative=0.05,reference_amplitude_relative=0.05,
                        reference_phase_deg=5.0),
        **capture_source(ROOT))
    (output/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    steps = int(round(args.cycles/(args.frequency*mp.lattice.dt_s)))
    summary['n_steps'] = steps
    print(f'{args.mode}, wall={args.wall}, nx={solver.nx}, ny={solver.ny}, steps={steps}, '
          f'delta/dx={reference.delta_T/mp.lattice.dx_m:.3f}',flush=True)
    began = time.perf_counter()
    coupled = None
    try:
        if args.mode == 'A':
            solver.initialize_from_macro(mp.lattice.rho_ref_lu,np.zeros(2),mp.theta_ref_lu)
            solver.observe()
            factory = make_symmetric_mass_neutral_wall_callback if args.wall=='mass_neutral' else make_bottom_grad_wall_callback
            for i in range(steps):
                ts = (i+1)*mp.lattice.dt_s
                Tw = ph.T0_K+args.amplitude*np.cos(2*np.pi*args.frequency*ts)
                solver.step(1,boundary_callback=factory(Tw/mp.temperature_scale,extrap='row1'))
        elif args.mode == 'B':
            solver.initialize_from_macro(mp.lattice.rho_ref_lu,np.zeros(2),mp.theta_ref_lu)
            solver.observe()
            half_c2=.5*np.sum(solver.lattice.c**2,axis=1)
            energy_scale=mp.pressure_scale*mp.lattice.dx_m
            # 独立通量夹具：把指定单侧区间热量施于同一双侧热带。
            for i in range(steps):
                ta,tb=i*mp.lattice.dt_s,(i+1)*mp.lattice.dt_s
                heat=args.amplitude*(np.cos(2*np.pi*args.frequency*ta)+
                    np.cos(2*np.pi*args.frequency*tb))*mp.lattice.dt_s
                def flux_wall(*,solver,f_post,g_post,f_stream,g_stream):
                    Erow=float(np.sum(f_stream[:1]*half_c2+g_stream[:1])/solver.nx*energy_scale)
                    rho_wall=float(np.mean(np.sum(f_stream[:1],axis=-1)))
                    capacity=.5*(mp.lattice.D+mp.lattice.S)*rho_wall*energy_scale/mp.temperature_scale
                    Tw=(Erow+heat)/capacity
                    return make_symmetric_mass_neutral_wall_callback(Tw/mp.temperature_scale,extrap='row1')(
                        solver=solver,f_post=f_post,g_post=g_post,f_stream=f_stream,g_stream=g_stream)
                solver.step(1,boundary_callback=flux_wall)
        else:
            coupled = run_levelc_predictor_corrector(solver=solver,
                params=FilmOdeParams(C_A_si=args.capacity,T_ref_K=ph.T0_K,gas_flux_factor=2),
                drive=SinusoidalDrive(0,complex(args.amplitude),args.frequency),n_steps=steps,
                wall_bc=args.wall,grad_extrap='row1',q_feedback_relax=args.relax,
                q_feedback_scale=1,q_extraction=None,probe=(10*args.refine,0),scheme=args.scheme)
        summary.update(analyze(solver,args,coupled,output))
        summary['run_completed'] = True
    except Exception as exc:
        summary.update(run_completed=False,error=str(exc),traceback=traceback.format_exc())
        if solver.samples:
            np.savez_compressed(output/'partial_observations.npz',**{
                k:np.asarray([s[k] for s in solver.samples]) for k in solver.samples[0]})
        print(summary['traceback'],flush=True)
    summary['elapsed_s'] = time.perf_counter()-began
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'诊断结束，run_completed={summary["run_completed"]}, elapsed={summary["elapsed_s"]:.2f}s',flush=True)
    if not summary['run_completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
