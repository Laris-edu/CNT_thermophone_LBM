"""Lim 2013 近/远场复现，以及本仓 LBM 链路对它的对照。

两种模式：

* ``--mode analytic``：用 ``reference/lim2013.py`` 复现论文 Fig. 2(a)/4/5 的曲线
  （近场 RMS 声压 vs 频率、vs 单位面积热容、vs 距离），并给出精确解与 Eq. (22)
  近似的偏离度——即论文 §4.4 自述的适用范围。
* ``--mode lbm``：在 10 kHz 工作点跑本仓保守耦合链路，用**壁面热流**（不是压力探针）
  经 ``p_hat = (gamma-1) q_hat / c0`` 换算近场声压，与 Lim 逐项对照。

为什么要走热流：本仓是封闭周期域，压力探针读的是封箱等容压缩
``p_box = 2(gamma-1)q/(Omega*H)``，在本工作点比辐射声压大 90 倍以上，不能当声压用。
热流则不受影响（域高 6.3 delta_T，热波在远端已衰减到 2e-3）。

对照时 Lim 侧一律用**本仓自己的气体常数**（``gas_from_physical_params``），
使残差只反映模型差异，不含常数口径差；论文图则用 Table 1 原值。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from config_io import load_config
from core.solver import GasSolver2D
from coupling.conjugate import run_levelc_predictor_corrector
from coupling.drive import SinusoidalDrive
from coupling.film_ode import FilmOdeParams
from postproc.multiharmonic_fit import fit_multiharmonic
from reference.constants import PhysicalParams
from reference.film_ode import level_c_closed_form
from reference.lim2013 import (
    TABLE1_AIR,
    TABLE1_ARGON,
    areal_drive_from_power,
    conductive_heat_flux_hat,
    exact_thermal_admittance,
    film_temperature_hat,
    gas_from_physical_params,
    near_field_pressure_hat_from_flux,
    near_field_pressure_rms_approximate,
    near_field_pressure_rms_exact,
    rayleigh_distance_m,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/gas_air_10k_d2q37_qmatched_local_transport.yaml"

# 论文 §4.1 的算例：3x3 cm^2 悬空单层膜，输入功率 0.85 W，空气中 beta0 = 16 W/(m^2 K)。
PAPER_AREA_M2 = 9.0e-4
PAPER_POWER_W = 0.85
PAPER_BETA0 = 16.0
PAPER_CS = 1.0e-3

_MIN_HARMONICS = 3
_DETREND_ORDER = 1


def _ratio(got: complex, expected: complex) -> tuple[float, float]:
    """返回 (相对幅值偏差, 相位偏差 deg)。"""

    if expected == 0.0:
        raise ValueError("reference amplitude is zero")
    return abs(got) / abs(expected) - 1.0, math.degrees(float(np.angle(got / expected)))


def _fit_harmonic(t: np.ndarray, x: np.ndarray, frequency_hz: float):
    return fit_multiharmonic(
        np.asarray(t, dtype=float),
        np.asarray(x, dtype=float),
        2.0 * math.pi * frequency_hz,
        n_harmonics=_MIN_HARMONICS,
        detrend_order=_DETREND_ORDER,
    )


def analytic_scan() -> dict:
    """论文 Fig. 2(a)/4/5 的曲线与本模块精确解的偏离度。"""

    frequencies = np.logspace(1.0, 6.0, 51)  # 10 Hz .. 1 MHz
    probe_x = 0.115  # 论文 Fig. 2 的测距
    for name, gas in (("air", TABLE1_AIR), ("argon", TABLE1_ARGON)):
        beta0 = PAPER_BETA0 if name == "air" else 15.0
        rows = []
        for f in frequencies:
            T_s = film_temperature_hat(
                f, -PAPER_POWER_W / PAPER_AREA_M2, PAPER_CS,
                beta0_per_side_W_m2K=beta0, gas=gas, exact=True,
            )
            q = conductive_heat_flux_hat(f, T_s, gas=gas, exact=True)
            approx = near_field_pressure_rms_approximate(
                f, PAPER_POWER_W, PAPER_AREA_M2, PAPER_CS,
                beta0_per_side_W_m2K=beta0, gas=gas,
            )
            # 精确解 = Eq. (9) 全解（热分支 + 声分支）；Eq. (24) 是它丢掉热分支后的形式。
            exact = near_field_pressure_rms_exact(f, T_s, x_m=probe_x, gas=gas)
            # 紧致热层恒等式 (gamma-1)q/c0，用于单独校验 Eq. (19)/(A.14)。
            from_flux = abs(near_field_pressure_hat_from_flux(q, gas)) / math.sqrt(2.0)
            rows.append(
                {
                    "f_Hz": float(f),
                    "p_rms_approx_Eq22_Pa": float(approx),
                    "p_rms_exact_Eq9_Pa": float(exact),
                    "p_rms_from_flux_identity_Pa": float(from_flux),
                    "exact_over_Eq22": float(exact / approx) if approx > 0.0 else None,
                    "exact_over_Eq22_at_x0": float(
                        near_field_pressure_rms_exact(f, T_s, x_m=0.0, gas=gas) / approx),
                    "flux_identity_over_Eq22": float(from_flux / approx) if approx > 0.0 else None,
                    "T_s_hat_K": complex(T_s),
                    "q_hat_one_sided_W_m2": complex(q),
                    "rayleigh_distance_m": rayleigh_distance_m(f, PAPER_AREA_M2, gas),
                }
            )
        yield name, gas, beta0, rows


def run_analytic(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"scope": "Lim 2013 解析复现（无 LBM）", "paper_case": {
        "area_m2": PAPER_AREA_M2, "power_W": PAPER_POWER_W,
        "beta0_per_side_W_m2K": PAPER_BETA0, "c_s_J_m2K": PAPER_CS}}

    gases: dict[str, dict] = {}
    for name, gas, beta0, rows in analytic_scan():
        band = [r for r in rows if 10.0 <= r["f_Hz"] <= 1.0e5]
        at_1mhz = max(rows, key=lambda r: r["f_Hz"])
        gases[name] = {
            "gas": asdict(gas),
            "beta0_per_side_W_m2K": beta0,
            "max_abs_deviation_Eq22_vs_exact_10Hz_100kHz": float(
                max(abs(r["exact_over_Eq22"] - 1.0) for r in band)
            ),
            "max_abs_deviation_flux_identity_10Hz_100kHz": float(
                max(abs(r["flux_identity_over_Eq22"] - 1.0) for r in band)
            ),
            "exact_over_Eq22_at_1MHz": at_1mhz["exact_over_Eq22"],
            "points": [{k: (v if not isinstance(v, complex) else [v.real, v.imag])
                        for k, v in r.items()} for r in rows],
        }
        print(f"[analytic] {name}: Eq.(22) vs 精确解 10 Hz-100 kHz 内最大偏离 "
              f"{100*gases[name]['max_abs_deviation_Eq22_vs_exact_10Hz_100kHz']:.3f}% ; "
              f"紧致热层恒等式偏离 "
              f"{100*gases[name]['max_abs_deviation_flux_identity_10Hz_100kHz']:.3f}% ; "
              f"1 MHz 处 exact/Eq22 = {gases[name]['exact_over_Eq22_at_1MHz']:.4f}")
    summary["gases"] = gases

    # Fig. 2：论文的对比算例，横轴 50-100 kHz，11 个点，空气与氩气各一幅。
    # SPL 约定：论文图上的 Eq.(22) 曲线与实验曲线在该频段几乎重合，只有在把 Eq.(22)
    # 的 P_rms 当作**幅值**（即 SPL = 20log10(P_rms/sqrt(2)/20uPa)）时才重合；
    # 若按字面 RMS 计算会高 3.01 dB、明显浮在实验点之上。两种都记录，供核对。
    fig2 = {}
    for name, gas, beta0 in (("air", TABLE1_AIR, PAPER_BETA0),
                             ("argon", TABLE1_ARGON, 15.0)):
        points = []
        for f in np.arange(50.0, 100.0 + 1e-9, 5.0) * 1.0e3:
            p = near_field_pressure_rms_approximate(
                float(f), PAPER_POWER_W, PAPER_AREA_M2, PAPER_CS,
                beta0_per_side_W_m2K=beta0, gas=gas)
            points.append({
                "f_Hz": float(f),
                "p_rms_Eq22_Pa": float(p),
                "spl_dB_treating_Eq22_as_rms": float(20.0 * math.log10(p / 20.0e-6)),
                "spl_dB_treating_Eq22_as_amplitude": float(
                    20.0 * math.log10(p / math.sqrt(2.0) / 20.0e-6)),
            })
        fig2[name] = {"beta0_per_side_W_m2K": beta0, "points": points}
        first, last = points[0], points[-1]
        print(f"[analytic] Fig.2({name}): 50 kHz {first['spl_dB_treating_Eq22_as_amplitude']:.2f} dB "
              f"-> 100 kHz {last['spl_dB_treating_Eq22_as_amplitude']:.2f} dB "
              f"(按幅值口径; 按 RMS 口径 +3.01 dB)")
    summary["fig2_paper_comparison"] = {
        "area_m2": PAPER_AREA_M2, "power_W": PAPER_POWER_W, "c_s_J_m2K": PAPER_CS,
        "panels": fig2,
    }

    # Fig. 4：近场声压 vs 单位面积热容（Pin=1 W, s=10 cm^2, 10 kHz）。
    area = 1.0e-3
    cs_scan = np.logspace(-4.0, 1.0, 41)
    fig4 = []
    for f in (1.0e4, 5.0e4, 1.0e5):
        fig4.append({
            "f_Hz": f,
            "points": [
                {"c_s_J_m2K": float(cs),
                 "p_rms_Pa": near_field_pressure_rms_approximate(
                     f, 1.0, area, float(cs), beta0_per_side_W_m2K=PAPER_BETA0, gas=TABLE1_AIR)}
                for cs in cs_scan
            ],
        })
    summary["fig4_heat_capacity"] = {"area_m2": area, "power_W": 1.0, "curves": fig4}

    # Fig. 5：近场 -> 远场随距离的转换（Pin=1 W, s=10 cm^2）。
    fig5 = []
    for f in (1.0e4, 5.0e4, 1.0e5):
        R0 = rayleigh_distance_m(f, area, TABLE1_AIR)
        near = near_field_pressure_rms_approximate(
            f, 1.0, area, PAPER_CS, beta0_per_side_W_m2K=PAPER_BETA0, gas=TABLE1_AIR)
        fig5.append({
            "f_Hz": f,
            "rayleigh_distance_m": R0,
            "p_near_rms_Pa": near,
            "p_at_0p1m_rms_Pa": near * R0 / 0.1,
            "far_field_applies_at_0p1m": bool(0.1 > R0),
        })
        print(f"[analytic] f={f/1e3:6.0f} kHz: R0 = {R0*1e3:7.3f} mm, 近场 {near:.4g} Pa, "
              f"0.1 m 处 {'远场' if 0.1 > R0 else '仍在近场'}")
    summary["fig5_distance"] = {"area_m2": area, "power_W": 1.0, "curves": fig5}

    (out_dir / "lim2013_analytic.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[analytic] 写入 {out_dir/'lim2013_analytic.json'}")
    return summary


def run_lbm(args: argparse.Namespace) -> dict:
    """10 kHz 工作点：LBM 壁面热流 -> 近场声压，与 Lim 逐项对照。"""

    cfg = load_config(args.config)
    cfg["numerics"].update(nx=args.nx, ny=args.ny)
    solver = GasSolver2D(cfg)
    mapping = solver.mapping
    phys = mapping.physical
    gas = gas_from_physical_params(PhysicalParams(
        rho0=phys.rho0_kg_m3, cp=phys.cp_J_kgK, kg=phys.kg_W_mK,
        c0=phys.c0_m_s, gamma=phys.gamma, p0=phys.p0_Pa, T0=phys.T0_K,
    ))
    c_s = args.capacity
    beta0 = args.beta0
    area = args.area
    power = args.power
    drive_areal = areal_drive_from_power(power, area)

    dt = float(mapping.lattice.dt_s)
    n_steps = int(round(args.cycles / (args.frequency * dt)))
    # Lim Eq. (1)：P_in(1 - cos Omega t)，复幅为 -P_in/s。这里先只取 AC 部分，
    # 避免封箱内 DC 加热无汇导致的漂移污染 AC 谐波（DC 效应另行扫描）。
    drive = SinusoidalDrive(0.0, complex(-drive_areal), args.frequency)

    params = FilmOdeParams(
        C_A_si=c_s, T_ref_K=gas.T0_K,
        linear_leak_conductance_si=2.0 * beta0, gas_flux_factor=2.0,
    )

    print(f"[lbm] nx={args.nx} ny={args.ny} steps={n_steps} "
          f"(delta_T/dx={math.sqrt(2*gas.alpha_m2_s/(2*math.pi*args.frequency))/mapping.lattice.dx_m:.1f}), "
          f"c_s={c_s:g}, beta0={beta0:g} (G={2*beta0:g}), drive={drive_areal:.1f} W/m^2")
    began = time.perf_counter()
    result = run_levelc_predictor_corrector(
        solver=solver, params=params, drive=drive, n_steps=n_steps,
        wall_bc="mass_neutral", grad_extrap="row1", probe=(min(10, solver.ny - 1), 0),
    )
    elapsed = time.perf_counter() - began

    t = result.t_si
    window = (t * args.frequency >= args.cycles - 1) & (t * args.frequency < args.cycles)
    t_mid = 0.5 * (t[1:] + t[:-1])
    q_mid_window = (t_mid * args.frequency >= args.cycles - 1) & (t_mid * args.frequency < args.cycles)

    fit_T = _fit_harmonic(t[window], result.T_s_K[window], args.frequency)
    fit_q = _fit_harmonic(t_mid[q_mid_window], result.q_g_one_sided_si[1:][q_mid_window], args.frequency)
    fit_p_box = _fit_harmonic(t[window], result.p_box_mean_Pa[window], args.frequency)

    T_lbm = fit_T.harmonic(1)
    q_lbm = fit_q.harmonic(1)

    # Lim 侧：同一气体常数、同一 c_s/beta0、同一面功率。
    T_lim = film_temperature_hat(
        args.frequency, -drive_areal, c_s, beta0_per_side_W_m2K=beta0, gas=gas, exact=True)
    q_lim = conductive_heat_flux_hat(args.frequency, T_lim, gas=gas, exact=True)
    M_exact = exact_thermal_admittance(args.frequency, gas)
    M_approx = math.sqrt(2.0 * math.pi * args.frequency / gas.alpha_m2_s) * complex(math.cos(math.pi / 4), math.sin(math.pi / 4))

    # 仓库自己的链路（同一气体常数，注意它内部已含 2*beta0 与 2*Y 因子）。
    repo_params = PhysicalParams(
        rho0=gas.rho0_kg_m3, cp=gas.cp_J_kgK, kg=gas.kappa_W_mK,
        c0=gas.c0_m_s, gamma=gas.gamma, p0=gas.p0_Pa, T0=gas.T0_K, beta0=beta0,
    )
    repo_closed = level_c_closed_form(
        args.frequency, complex(-drive_areal), C_A=c_s, params=repo_params)

    p_from_q_lbm = near_field_pressure_hat_from_flux(q_lbm, gas)
    p_from_q_lim = near_field_pressure_hat_from_flux(q_lim, gas)
    p_eq22 = near_field_pressure_rms_approximate(
        args.frequency, power, area, c_s, beta0_per_side_W_m2K=beta0, gas=gas)

    delta_T = math.sqrt(2.0 * gas.alpha_m2_s / (2.0 * math.pi * args.frequency))
    height = solver.ny * mapping.lattice.dx_m
    ratio_box = 2.0 * (gas.gamma - 1.0) * abs(q_lbm) / (2.0 * math.pi * args.frequency * height)

    rows = []
    for name, got, expected in (
        ("q_g_one_sided_W_m2: LBM vs Lim exact", q_lbm, q_lim),
        ("q_g_one_sided_W_m2: LBM vs repo closed form", q_lbm, repo_closed.q_g_hat),
        ("T_s_hat_K: LBM vs Lim exact", T_lbm, T_lim),
        ("T_s_hat_K: LBM vs repo closed form", T_lbm, repo_closed.T_s_hat),
        ("p_hat_near_Pa: LBM(q) vs Lim(q)", p_from_q_lbm, p_from_q_lim),
    ):
        da, dp = _ratio(got, expected)
        rows.append({"quantity": name, "lbm": complex(got), "reference": complex(expected),
                     "amplitude_rel": float(da), "phase_deg": float(dp)})
        print(f"  {name:44s} dAmp={100*da:+7.2f}%  dPhase={dp:+7.2f} deg")

    return {
        "scope": "10 kHz 近场：LBM 壁面热流 -> 声压，对照 Lim 2013",
        "case": {
            "frequency_Hz": args.frequency, "cycles": args.cycles, "n_steps": n_steps,
            "nx": args.nx, "ny": args.ny, "dx_m": mapping.lattice.dx_m, "dt_s": dt,
            "c_s_J_m2K": c_s, "beta0_per_side_W_m2K": beta0,
            "area_m2": area, "power_W": power, "drive_areal_W_m2": drive_areal,
            "delta_T_m": delta_T, "delta_T_over_dx": delta_T / mapping.lattice.dx_m,
            "height_m": height, "height_over_delta_T": height / delta_T,
            "rayleigh_distance_m": rayleigh_distance_m(args.frequency, area, gas),
        },
        "gas": asdict(gas),
        "lbm": {
            "T_s_hat_K": [T_lbm.real, T_lbm.imag],
            "q_g_hat_one_sided_W_m2": [q_lbm.real, q_lbm.imag],
            "p_box_mean_hat_Pa": [fit_p_box.harmonic(1).real, fit_p_box.harmonic(1).imag],
            "p_near_from_q_hat_Pa": [p_from_q_lbm.real, p_from_q_lbm.imag],
            "T_s_fit_residual_rms_K": fit_T.residual_rms,
            "q_fit_residual_rms_W_m2": fit_q.residual_rms,
            "T_s_trend_coeffs_raw_K": [float(v) for v in fit_T.trend_coeffs_raw],
            "film_energy_audit_relative": result.energy_audit.max_relative_residual,
            "total_energy_audit_relative": result.total_energy_audit.max_relative_residual,
            "mass_relative_drift": float(np.max(np.abs(result.mass_lu / result.mass_lu[0] - 1.0))),
            "max_wall_vs_film_K": float(np.max(np.abs(result.wall_temperature_error_K))),
            "finite": result.finite,
        },
        "lim_exact": {
            "T_s_hat_K": [T_lim.real, T_lim.imag],
            "q_hat_one_sided_W_m2": [q_lim.real, q_lim.imag],
            "M_exact": [M_exact.real, M_exact.imag],
            "M_approx_sqrt_jOmega_over_alpha": [M_approx.real, M_approx.imag],
            "p_rms_Eq22_Pa": p_eq22,
        },
        "repo_closed_form": {
            "T_s_hat_K": [repo_closed.T_s_hat.real, repo_closed.T_s_hat.imag],
            "q_g_hat_W_m2": [repo_closed.q_g_hat.real, repo_closed.q_g_hat.imag],
        },
        "closed_box_artifact": {
            "p_box_quasistatic_amplitude_Pa": ratio_box,
            "p_box_over_radiation": ratio_box / p_eq22,
            "delta_p_over_p0": ratio_box / gas.p0_Pa,
            "note": "本仓域是封闭周期箱，压力探针读等容压缩，不可当声压；声压必须由 q_g 换算",
        },
        "comparisons": [
            {**r, "lbm": [r["lbm"].real, r["lbm"].imag],
             "reference": [r["reference"].real, r["reference"].imag]}
            for r in rows
        ],
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["analytic", "lbm", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--frequency", type=float, default=1.0e4)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--capacity", type=float, default=PAPER_CS,
                        help="c_s，单位面积热容 J/(m^2 K)；论文提到 CNT 膜量级 1e-3")
    parser.add_argument("--beta0", type=float, default=PAPER_BETA0,
                        help="单侧热损系数 W/(m^2 K)；链路里按 G=2*beta0 使用")
    parser.add_argument("--area", type=float, default=PAPER_AREA_M2)
    parser.add_argument("--power", type=float, default=PAPER_POWER_W)
    args = parser.parse_args()
    if args.frequency <= 0.0 or args.cycles < 3 or args.ny < 24 or args.nx < 1:
        parser.error("需要 frequency>0, cycles>=3, ny>=24, nx>=1")
    if args.capacity <= 0.0 or args.beta0 < 0.0 or args.area <= 0.0 or args.power <= 0.0:
        parser.error("capacity/area/power 必须为正，beta0 非负")

    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.mode in ("analytic", "both"):
        summary["analytic"] = run_analytic(args.output)
    if args.mode in ("lbm", "both"):
        summary["lbm"] = run_lbm(args)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"写入 {args.output/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
