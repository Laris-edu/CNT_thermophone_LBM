"""Compute axial SPL from a saved strip configuration and surface phasors.

Reuses the public Lim, unit mapping and 2D Kirchhoff interfaces. The numerical
curve remains an unvalidated infinite-z strip. Literature points are reported
values for a finite square, not measurements of this configured material.
An optional analytical-only heat capacity and global SPL level fit preserve
the original LBM data and never represent a new LBM material calculation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from farfield.kirchhoff_2d import GREEN_OUTGOING, dpdn_from_velocity, kirchhoff_2d_frequency
from farfield.strip_surface import mirror_strip_surface
from reference.lim2013 import matched_gas, rayleigh_distance_m, solve_lim2013_frequency
from simulation.provenance import capture_source


def spl(pressure_rms):
    values = np.asarray(pressure_rms)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("SPL requires finite positive RMS acoustic pressure")
    return 20 * np.log10(values / 20e-6)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--batch", required=True, help="New batch directly under Compute/lim2013")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qa-dir", type=Path, help="Directory containing audit_panel_alignment.py")
    parser.add_argument("--analytical-capacity", type=float, help="Override only Lim C_A, in J/(m² K)")
    parser.add_argument("--fit-existing-lbm", action="store_true", help="Fit one global LBM SPL offset at the five saved distances")
    args = parser.parse_args()
    topic = Path(__file__).resolve().parent
    root = topic.parents[1]
    output = (topic / args.batch).resolve()
    if output.parent != topic or output.exists():
        raise ValueError("Use a new batch name directly under Compute/lim2013")
    source = args.source_batch.resolve()
    paths = dict(config=source / "raw/power_r2/effective.yaml",
                 run_summary=source / "raw/power_r2/summary.json",
                 run_manifest=source / "raw/power_r2/manifest.json",
                 nearfield=source / "final/power_r2_nearfield.npz",
                 nearfield_summary=source / "final/power_r2_nearfield.json",
                 saved_axial=source / "final/axial_spl_6_8_15_40_45mm_kirchhoff.json",
                 propagation=source / "final/propagation_unvalidated.json",
                 paper=root / "Paper/LI.pdf/full.md")
    cfg = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    saved = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in
             ("run_summary", "run_manifest", "nearfield_summary", "saved_axial", "propagation")}
    run = cfg["strip"]
    if run["kind"] != "power" or saved["run_summary"]["status"] != "completed":
        raise ValueError("A completed harmonic power case is required")
    gas = matched_gas(create_unit_mapping(cfg))
    f, area = float(run["frequency_Hz"]), float(cfg["material"]["nominal_projected_area_m2"])
    source_capacity = float(run["C_A_J_m2K"])
    capacity = source_capacity if args.analytical_capacity is None else args.analytical_capacity
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("Analytical heat capacity must be finite and positive")
    drive, beta = -float(run["mean_power_W_m2"]), float(run["G_W_m2K"]) / 2
    power = -drive * area
    radius = rayleigh_distance_m(f, area, gas)
    text = paths["paper"].read_text(encoding="utf-8").split("表4.1 数值预测与实验数据之间的误差", 1)[1]
    table = ET.fromstring(re.search(r"<table>.*?</table>", text, flags=re.S).group())
    paper_rows = [[float(c.text) for c in row] for row in list(table)[1:]]
    # Exclude only the five 10 kHz rows: this configured case has sound frequency 20 kHz.
    experiment = np.array([row[:3] for row in paper_rows if np.isclose(row[0] * 1000, f)])
    if experiment.shape != (5, 3):
        raise ValueError("Expected the five frequency-matched literature observations")
    points = experiment[:, 1] * .001
    if not np.isclose(power, cfg["material"]["nominal_mean_total_power_W"]):
        raise ValueError("Projected-area power normalization is inconsistent")
    if not np.isclose(saved["nearfield_summary"]["frequency_Hz"], f):
        raise ValueError("Saved surface phasors have a different frequency")
    preview = dict(source=str(source), output=str(output), frequency_Hz=f, power_W=power,
                   analytical_C_A_J_m2K=capacity, saved_LBM_C_A_J_m2K=source_capacity,
                   empirical_fit_requested=args.fit_existing_lbm,
                   beta0_per_side_W_m2K=beta, gas=asdict(gas),
                   rayleigh_distance_mm=radius * 1000, distance_range_mm=[.5, 100],
                   time_steps=0, new_time_domain_simulations=0)
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if args.qa_dir is None:
        raise ValueError("Supply --qa-dir for the rendered plot geometry check")
    sys.path.insert(0, str(args.qa_dir.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    started = datetime.now(timezone.utc).isoformat()
    distances = np.unique(np.r_[np.linspace(.0005, .1, 401), points, radius])
    results = [solve_lim2013_frequency(f, drive, capacity, area_m2=area, x_m=float(y),
               beta0_per_side_W_m2K=beta, gas=gas) for y in distances]
    lim_rms = np.array([r.p_rms_near_approximate_Pa if r.in_near_field else r.p_rms_far_Pa for r in results])
    exact_hat = np.array([r.p_hat_acoustic_exact_Pa for r in results])
    with np.load(paths["nearfield"], allow_pickle=False) as z:
        contour = saved["nearfield_summary"]["selected_export_contour"]
        a, b = z["surface_slices"][contour - 1]
        upper = {key: z["surface_" + key][a:b] for key in ("x_m", "y_m", "normal_x", "normal_y", "ds_m")}
        closed = mirror_strip_surface(upper, z["surface_p_hat_Pa"][a:b],
                                      z["surface_ux_hat_m_s"][a:b], z["surface_uy_hat_m_s"][a:b])
    if distances.min() <= upper["y_m"].max():
        raise ValueError("All propagated observers must lie above the closed source surface")
    omega = 2 * np.pi * f
    dpdn = dpdn_from_velocity(closed["un_hat_m_s"], omega_rad_s=omega, rho0_kg_m3=gas.rho0_kg_m3)
    numerical_hat = np.empty(len(distances), dtype=complex)
    for start in range(0, len(distances), 16):
        ys = distances[start:start + 16]
        numerical_hat[start:start + 16] = kirchhoff_2d_frequency(
            surface_x_m=closed["x_m"], surface_y_m=closed["y_m"],
            normal_x=closed["normal_x"], normal_y=closed["normal_y"], ds_m=closed["ds_m"],
            p_hat_Pa=closed["p_hat_Pa"], dpdn_hat_Pa_m=dpdn,
            observer_x_m=np.zeros(len(ys)), observer_y_m=ys, omega_rad_s=omega,
            c0_m_s=gas.c0_m_s, green_convention=GREEN_OUTGOING)
    numerical_rms = np.abs(numerical_hat) / np.sqrt(2)
    lim_spl, numerical_spl = spl(lim_rms), spl(numerical_rms)
    indices = np.array([np.flatnonzero(distances == p)[0] for p in points])
    old = saved["saved_axial"]["points"]
    old_hat = np.array([complex(p["pressure_hat_real_Pa"], p["pressure_hat_imag_Pa"]) for p in old])
    if not np.allclose([p["distance_mm"] for p in old], experiment[:, 1], rtol=0, atol=1e-10):
        raise ValueError("Saved axial points do not match literature distances")
    reuse_error = float(np.max(np.abs(numerical_hat[indices] - old_hat)))
    if reuse_error > 1e-12:
        raise ValueError(f"Public propagation did not reproduce saved phasors: {reuse_error}")
    fit_info = None
    if args.fit_existing_lbm:
        # Equal-weight least squares in dB, with one scalar fitted on the five
        # saved distances. Literature SPL is not used to estimate this offset.
        offset = float(np.mean(lim_spl[indices] - numerical_spl[indices]))
        pressure_factor = float(10 ** (offset / 20))
        fitted_spl = numerical_spl + offset
        residual = fitted_spl[indices] - lim_spl[indices]
        fit_info = dict(method="One global additive SPL offset; equal-weight least squares in dB",
                        equation="L_empirical(y) = L_saved_LBM(y) + offset_dB",
                        target=f"Lim approximation with C_A={capacity:g} J/(m² K)",
                        fit_distances_mm=(points * 1000).tolist(), offset_dB=offset,
                        pressure_amplitude_factor=pressure_factor, fit_point_count=len(indices),
                        fit_point_RMSE_dB=float(np.sqrt(np.mean(residual ** 2))),
                        fit_point_max_abs_residual_dB=float(np.max(abs(residual))),
                        fit_point_mean_residual_dB=float(np.mean(residual)),
                        residual_dB=residual.tolist(), literature_used_as_fit_target=False,
                        limitation="Five training points only; outside 6–45 mm is empirical extrapolation. No LBM rerun, physical heat-capacity update, or independent validation.")
    rows = [dict(distance_mm=float(distances[i] * 1000),
                 Lim_equation="22" if distances[i] <= radius else "23",
                 Lim_p_rms_Pa=float(lim_rms[i]), Lim_SPL_dB=float(lim_spl[i]),
                 LBM_Kirchhoff_p_rms_Pa=float(numerical_rms[i]), LBM_Kirchhoff_SPL_dB=float(numerical_spl[i]))
            for i in range(len(distances))]
    if fit_info is not None:
        for i, row in enumerate(rows):
            row.update(LBM_empirical_level_fit_p_rms_Pa=float(numerical_rms[i] * pressure_factor),
                       LBM_empirical_level_fit_SPL_dB=float(fitted_spl[i]),
                       empirical_fit_minus_Lim_dB=float(fitted_spl[i] - lim_spl[i]))
    selected = [{**rows[i], "LI_reported_experiment_SPL_dB": float(exp[2])} for i, exp in zip(indices, experiment)]
    raw, final = output / "raw/axial_spl", output / "final"
    raw.mkdir(parents=True, exist_ok=False)
    final.mkdir()
    for name, values in (("spl_vs_distance.csv", rows), ("spl_at_paper_distances.csv", selected)):
        with (final / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    np.savez_compressed(raw / "trajectory.npz", distance_m=distances, Lim_p_rms_Pa=lim_rms,
                        Lim_acoustic_exact_hat_Pa=exact_hat, LBM_Kirchhoff_hat_Pa=numerical_hat,
                        literature_distance_m=points, literature_reported_SPL_dB=experiment[:, 2],
                        LBM_empirical_level_fit_SPL_dB=np.array([]) if fit_info is None else fitted_spl)
    effective = {**cfg, "spl_comparison": {**preview, "pressure_reference_Pa": 20e-6,
                 "selected_surface_contour": contour, "literature_source": "LI, Table 4.1, 20 kHz"}}
    (raw / "effective.yaml").write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = dict(status="completed", validation_passed=None,
                   source_run_validation_passed=saved["run_summary"]["validation_passed"],
                   source_validation_passed=saved["nearfield_summary"]["source_validation_passed"],
                   source_propagation_validation_passed=saved["propagation"]["propagation_validation_passed"],
                   numerical_checks=dict(finite_positive_RMS=True, finite_SPL=True,
                       saved_axial_max_complex_difference_Pa=reuse_error,
                       literature_rows_before=len(paper_rows), literature_rows_after=len(experiment)),
                   SPL_definition="Raw SPL = 20log10(abs(peak phasor)/sqrt(2)/(20e-6 Pa)); original complex phasors retained. Any empirical level offset is stored separately.",
                   analytical_C_A_J_m2K=capacity, saved_LBM_C_A_J_m2K=source_capacity,
                   empirical_fit=fit_info,
                   nearfield_SPL_dB=float(lim_spl[0]), rayleigh_distance_mm=radius * 1000,
                   points=selected, fit_window=saved["nearfield_summary"]["fit_window"],
                   scope="Configured 2D infinite-z LBM strip versus Lim planar/spherical approximation. Literature square-film thermal properties and SPL convention are not fully specified; reported values are not adjusted.")
    provenance = capture_source(root)
    manifest = dict(run_state="completed", run_kind="saved_data_SPL_postprocessing",
                    started_UTC=started, ended_UTC=datetime.now(timezone.utc).isoformat(), **provenance,
                    new_time_domain_simulations=0, input_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
                    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    analysis_script_git_status=subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", str(Path(__file__))], text=True).strip(),
                    source_run_git_commit=saved["run_manifest"].get("git_commit"),
                    source_run_git_dirty=saved["run_manifest"].get("git_dirty"))
    # Single comparison panel; deterministic model curves and reported observations,
    # with no invented experimental error bars or statistical inference.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2047244094, 4.5275590551))  # 183 by 115 mm
    fig.subplots_adjust(left=.12, right=.97, bottom=.29 if fit_info else .26, top=.75 if fit_info else .87)
    ax.plot(distances * 1000, lim_spl, color="#C44E52", lw=1.7, ls="--",
            label=f"Lim, C_A = {capacity:g}" if fit_info else "Lim planar / spherical approximation")
    ax.plot(distances * 1000, numerical_spl, color="#999999" if fit_info else "#D89000", lw=1.2 if fit_info else 1.7,
            label=f"Saved LBM, C_A = {source_capacity:g}" if fit_info else "2D LBM + Kirchhoff (unvalidated)")
    if fit_info is not None:
        ax.plot(distances * 1000, fitted_spl, color="#D89000", lw=1.7, label="LBM + fitted global level offset")
        ax.scatter(points * 1000, fitted_spl[indices], s=20, color="#D89000", edgecolors="none", zorder=4)
    ax.scatter(experiment[:, 1], experiment[:, 2], s=29, facecolors="none", edgecolors="#222222", lw=1.1,
               label="LI reported experiment" if fit_info else "LI experiment, Table 4.1 (finite square)", zorder=4)
    ax.axvline(radius * 1000, color="#A0A0A0", lw=.8, ls=":")
    ax.text(radius * 1000 + 2, 69 if fit_info else 63.5, f"Rayleigh distance\n{radius * 1000:.2f} mm", color="#666666", fontsize=8)
    ax.set(xlabel="Distance from film centre along normal axis (mm)",
           ylabel="Sound pressure level (dB re 20 µPa)", xlim=(0, 100), ylim=(50, 94))
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(50, 95, 10))
    if fit_info is not None:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.05), ncol=2, frameon=False)
        fig.suptitle(f"Empirical SPL fit at {f / 1000:g} kHz", x=.12, ha="left", y=.97, fontsize=12)
        fig.text(.12, .905, f"P = {power:.2f} W; nominal area = {area * 1e4:g} cm²; Lim C_A = {capacity:g} J/(m² K); G = {2 * beta:g}", fontsize=8.5)
        fig.text(.12, .115, f"Global offset: {offset:.3f} dB; pressure factor: {pressure_factor:.6f}; fit-point RMSE: {fit_info['fit_point_RMSE_dB']:.3f} dB.", fontsize=8)
        fig.text(.12, .072, f"Original LBM retained. Level adjustment only; no new LBM calculation with C_A = {capacity:g}.", fontsize=8)
        fig.text(.12, .029, "2D infinite strip; literature finite-square values shown as reported. No new physical validation.", fontsize=8)
    else:
        ax.legend(loc="upper right", frameon=False)
        fig.suptitle(f"Axial SPL comparison at {f / 1000:g} kHz", x=.12, ha="left", y=.97, fontsize=12)
        fig.text(.12, .905, f"P = {power:.2f} W; nominal area = {area * 1e4:g} cm²; Lim C_A = {capacity:g}; saved LBM C_A = {source_capacity:g} J/(m² K)", fontsize=8.5)
        fig.text(.12, .09, "Numerical field: infinite-z strip. Literature: 30 × 30 mm finite square, unmatched film properties.", fontsize=8)
        fig.text(.12, .045, "Simulation curves use fundamental RMS; literature SPL values are shown as reported.", fontsize=8)
    work = output / "_work"
    work.mkdir()
    require_matplotlib_panel_alignment(fig, json_out=work / "spl_comparison.alignment.json", strict=True)
    fig.savefig(final / "spl_comparison.pdf")
    fig.savefig(final / "spl_comparison.svg")
    fig.savefig(final / "spl_comparison.png", dpi=600)
    plt.close(fig)
    manifest["ended_UTC"] = datetime.now(timezone.utc).isoformat()
    (raw / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(final), checks=summary["numerical_checks"], empirical_fit=fit_info, points=selected), indent=2))


if __name__ == "__main__":
    main()
