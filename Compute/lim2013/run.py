"""E1 工况的 Lim 频域计算；从根目录 python -m Compute.lim2013.run 执行。"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import yaml

from reference.lim2013 import (
    LimGas, areal_drive_from_power, rayleigh_distance_m, solve_lim2013_frequency,
)
from simulation.provenance import capture_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="不存在的新批次名")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    topic = Path(__file__).resolve().parent
    root = topic.parents[1]
    batch = (topic / args.batch).resolve()
    if batch.parent != topic or batch.exists():
        raise ValueError("batch must name a new directory directly under Compute/lim2013")
    study = topic / "study.yaml"
    cfg = yaml.safe_load(study.read_text(encoding="utf-8"))
    gas = LimGas(**cfg["gas"])
    frequency = float(cfg["frequency_Hz"])
    area = float(cfg["area_m2"])
    power = float(cfg["power_W"])
    distances = [float(x) for x in cfg["distances_m"]]
    beta = float(cfg["beta0_per_side_W_m2K"])
    reference_pressure = float(cfg["pressure_reference_Pa"])
    for value in (frequency, area, power, reference_pressure, *distances):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("frequency, area, power, reference pressure and distances must be positive")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta0 must be finite and non-negative")
    radius = rayleigh_distance_m(frequency, area, gas)
    capacity = cfg["c_s_J_m2K"]
    preview = dict(
        state="ready" if capacity is not None else "missing_film_heat_capacity",
        frequency_Hz=frequency, distances_m=distances,
        rayleigh_distance_m=radius, all_points_in_near_field=all(x <= radius for x in distances),
        drive_areal_hat_W_m2=-areal_drive_from_power(power, area),
        c_s_J_m2K=capacity, output=str(batch), time_steps=0,
        calculation="Lim frequency-domain analytic reference; no LBM time stepping",
    )
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return
    if capacity is None:
        raise ValueError("C_s is missing in the source table; provide the user-approved value in study.yaml")
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for distance in distances:
        result = solve_lim2013_frequency(
            frequency, preview["drive_areal_hat_W_m2"], float(capacity),
            area_m2=area, x_m=distance, beta0_per_side_W_m2K=beta, gas=gas,
        )
        pressure = result.p_rms_near_approximate_Pa if result.in_near_field else result.p_rms_far_Pa
        rows.append(dict(
            frequency_Hz=frequency, distance_mm=distance * 1000,
            field="near" if result.in_near_field else "far",
            equation="Lim Eq. (22)" if result.in_near_field else "Lim Eq. (23)",
            p_rms_Pa=pressure, SPL_dB=20 * math.log10(pressure / reference_pressure),
            rayleigh_distance_mm=result.rayleigh_distance_m * 1000,
            delta_T_um=result.delta_T_m * 1e6,
            x_over_delta_T=distance / result.delta_T_m,
            acoustic_decay_exponent=result.acoustic_decay_exponent,
            p_rms_planar_exact_Pa=result.p_rms_planar_exact_Pa,
            SPL_planar_exact_dB=20 * math.log10(result.p_rms_planar_exact_Pa / reference_pressure),
        ))
    checks = dict(
        finite_positive_pressures=all(math.isfinite(row["p_rms_Pa"]) and row["p_rms_Pa"] > 0 for row in rows),
        finite_SPL=all(math.isfinite(row["SPL_dB"]) for row in rows),
        all_points_in_near_field=all(row["field"] == "near" for row in rows),
        thermal_layer_excluded=min(row["x_over_delta_T"] for row in rows) > 10,
        omega_diffusion_ratio=2 * math.pi * frequency * gas.rho0_kg_m3 * gas.alpha_m2_s / gas.p0_Pa,
        k_delta_T=2 * math.pi * frequency / gas.c0_m_s * rows[0]["delta_T_um"] * 1e-6,
        near_SPL_span_dB=max(row["SPL_dB"] for row in rows) - min(row["SPL_dB"] for row in rows),
        maximum_acoustic_decay_exponent=max(row["acoustic_decay_exponent"] for row in rows),
    )
    if not (checks["finite_positive_pressures"] and checks["finite_SPL"]):
        raise ValueError("nonfinite response; no result batch written")
    source = capture_source(root)
    source_table = (topic / cfg["source_table"]).resolve()
    case_files = [Path(__file__).resolve(), study, source_table]
    manifest = dict(
        run_state="completed", run_kind="analytic_reference",
        started_UTC=started, ended_UTC=datetime.now(timezone.utc).isoformat(),
        **source,
        case_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in case_files},
        scope="Linear Lim reference; not finite-square BEM, LBM or experimental validation",
        convention="Re[hat*exp(+i*omega*t)]; RMS=abs(hat)/sqrt(2); heat flux is one-sided",
    )
    raw = batch / "raw" / "e1_20khz"
    final = batch / "final"
    batch.mkdir()
    raw.mkdir(parents=True)
    final.mkdir()
    (raw / "effective.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (raw / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(
        run_state="completed", device_validation_passed=None,
        validation_scope="No device validation performed; numerical applicability diagnostics only",
        numerical_checks=checks,
        c_s_source="User-supplied; absent from the parameter reference table",
        results_csv="../../final/spl_20khz.csv",
        limitations=[
            "The five requested points use the plane-wave near-field approximation, with no diffraction structure.",
            "The exact plane solution is a separate diagnostic and does not model a finite square film.",
            "Reported gas constants are preserved even though alpha and the ideal-gas identities are not fully consistent.",
        ],
    )
    (raw / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (final / "spl_20khz.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(output=str(final / "spl_20khz.csv"), rows=rows, checks=checks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
