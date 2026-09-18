"""P4-4 Kirchhoff kernel manufactured-fixture verification (contract section 9).

Runs the contract's four fixture families on farfield/kirchhoff_2d.py from
configs/phase4_kirchhoff_fixture.yaml and reports gate verdicts (kernel-level thresholds:
amplitude < 2%, phase < 2 deg). Pure Helmholtz, no LBM. The Green convention/prefactor is
anchored HERE (and in verification/test_phase4_kirchhoff.py which reuses these builders) and
is never re-tuned against end-to-end thermoacoustic results.

Fixtures (contract section 9.3):
  1 cylindrical: manufactured outgoing cylindrical wave (source below the line) sampled as
    (p_hat, dp_hat/dn) on the control line, extrapolated to observers, compared to the exact
    field -- pins the integral, the H0^(2) kind and the (-i/4) prefactor together;
  2 convergence: the same fixture at coarser surface samplings -- error decreases with
    refinement (the discretization is the error, not a lucky cancellation);
  3 phase convention: outgoing PLANE wave fed via the VELOCITY channel
    (dp/dn = -i Omega rho0 v_n) -- certifies the exp(+i Omega t) <-> H^(2) pairing on the
    exact input channel the D3-4 chain uses, observer phase must be -k h;
  4 prefactor lock + counterexample: amplitude ratio pinned at 1 (locks |prefactor|); the
    hankel1 kernel under this time convention must FAIL reconstruction by O(1)
    (discriminating power -- the fixture can actually see a wrong convention)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import hankel2

from farfield.kirchhoff_2d import (
    GREEN_OUTGOING,
    GREEN_WRONG_TIME_CONVENTION,
    dpdn_from_velocity,
    kirchhoff_2d_frequency,
)
from config_io import load_config

FIXTURE_CONFIG = Path("configs/phase4_kirchhoff_fixture.yaml")


def build_surface(cfg: dict, samples_per_wavelength: float | None = None):
    c0 = float(cfg["physical"]["c0_m_s"]); f = float(cfg["physical"]["frequency_Hz"])
    lam = c0 / f
    spw = float(samples_per_wavelength or cfg["surface"]["samples_per_wavelength"])
    L = float(cfg["surface"]["aperture_wavelengths"]) * lam
    n = int(round(L / lam * spw))
    xs = np.linspace(-L / 2.0, L / 2.0, n)
    return {
        "lam": lam, "k": 2.0 * math.pi * f / c0, "omega": 2.0 * math.pi * f, "c0": c0,
        "xs": xs, "ys": np.zeros(n), "nx": np.zeros(n), "ny": np.ones(n),
        "ds": np.full(n, xs[1] - xs[0]),
    }


def cylindrical_exact(cfg: dict, s: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = float(cfg["cylindrical_source"]["amplitude_Pa"])
    d = float(cfg["cylindrical_source"]["depth_wavelengths"]) * s["lam"]
    r = np.sqrt(np.asarray(x) ** 2 + (np.asarray(y) + d) ** 2)
    return A * (-0.25j) * hankel2(0, s["k"] * r)


def run_cylindrical(cfg: dict, samples_per_wavelength: float | None = None) -> dict[str, Any]:
    """Fixtures 1/2/4: cylindrical manufactured field -> observers; worst amp/phase error."""

    s = build_surface(cfg, samples_per_wavelength)
    A = float(cfg["cylindrical_source"]["amplitude_Pa"])
    d = float(cfg["cylindrical_source"]["depth_wavelengths"]) * s["lam"]
    rs = np.sqrt(s["xs"] ** 2 + d ** 2)
    p_surf = A * (-0.25j) * hankel2(0, s["k"] * rs)
    dpdn_surf = A * (-0.25j) * s["k"] * (-hankel2(1, s["k"] * rs)) * (d / rs)
    obs = np.asarray(cfg["observers_wavelengths"], dtype=float) * s["lam"]
    pk = kirchhoff_2d_frequency(
        surface_x_m=s["xs"], surface_y_m=s["ys"], normal_x=s["nx"], normal_y=s["ny"],
        ds_m=s["ds"], p_hat_Pa=p_surf, dpdn_hat_Pa_m=dpdn_surf,
        observer_x_m=obs[:, 0], observer_y_m=obs[:, 1],
        omega_rad_s=s["omega"], c0_m_s=s["c0"], green_convention=GREEN_OUTGOING)
    pe = cylindrical_exact(cfg, s, obs[:, 0], obs[:, 1])
    ratio = pk / pe
    return {
        "amp_err_max": float(np.max(np.abs(np.abs(ratio) - 1.0))),
        "phase_err_deg_max": float(np.max(np.abs(np.degrees(np.angle(ratio))))),
        "ratios": [[abs(r), math.degrees(np.angle(r))] for r in ratio],
    }


def run_plane_wave(cfg: dict) -> dict[str, Any]:
    """Fixture 3: plane wave via the velocity channel; phase must transport as -k h."""

    s = build_surface(cfg)
    rho0 = float(cfg["physical"]["rho0_kg_m3"])
    p0 = complex(float(cfg["plane_wave"]["amplitude_Pa"]))
    h = float(cfg["plane_wave"]["observer_height_wavelengths"]) * s["lam"]
    n = s["xs"].size
    v_n = np.full(n, p0 / (rho0 * s["c0"]))
    pk = kirchhoff_2d_frequency(
        surface_x_m=s["xs"], surface_y_m=s["ys"], normal_x=s["nx"], normal_y=s["ny"],
        ds_m=s["ds"], p_hat_Pa=np.full(n, p0),
        dpdn_hat_Pa_m=dpdn_from_velocity(v_n, omega_rad_s=s["omega"], rho0_kg_m3=rho0),
        observer_x_m=np.array([0.0]), observer_y_m=np.array([h]),
        omega_rad_s=s["omega"], c0_m_s=s["c0"], green_convention=GREEN_OUTGOING)
    ratio = complex(pk[0] / (p0 * np.exp(-1j * s["k"] * h)))
    return {"amp_err": abs(abs(ratio) - 1.0), "phase_err_deg": abs(math.degrees(np.angle(ratio)))}


def run_counterexample(cfg: dict) -> dict[str, Any]:
    """Fixture 4b: the hankel1 kernel under exp(+i Omega t) must fail by O(1)."""

    s = build_surface(cfg)
    A = float(cfg["cylindrical_source"]["amplitude_Pa"])
    d = float(cfg["cylindrical_source"]["depth_wavelengths"]) * s["lam"]
    rs = np.sqrt(s["xs"] ** 2 + d ** 2)
    p_surf = A * (-0.25j) * hankel2(0, s["k"] * rs)
    dpdn_surf = A * (-0.25j) * s["k"] * (-hankel2(1, s["k"] * rs)) * (d / rs)
    obs = np.asarray(cfg["observers_wavelengths"], dtype=float)[0] * s["lam"]
    pk = kirchhoff_2d_frequency(
        surface_x_m=s["xs"], surface_y_m=s["ys"], normal_x=s["nx"], normal_y=s["ny"],
        ds_m=s["ds"], p_hat_Pa=p_surf, dpdn_hat_Pa_m=dpdn_surf,
        observer_x_m=np.array([obs[0]]), observer_y_m=np.array([obs[1]]),
        omega_rad_s=s["omega"], c0_m_s=s["c0"],
        green_convention=GREEN_WRONG_TIME_CONVENTION)
    pe = cylindrical_exact(cfg, s, np.array([obs[0]]), np.array([obs[1]]))
    return {"reconstruction_error": float(abs(pk[0] / pe[0] - 1.0))}


def run_convergence(cfg: dict) -> dict[str, Any]:
    """Fixture 2: worst cylindrical error at increasing samples/wavelength."""

    errs = []
    for spw in cfg["convergence_samples_per_wavelength"]:
        r = run_cylindrical(cfg, samples_per_wavelength=float(spw))
        errs.append({"samples_per_wavelength": float(spw),
                     "amp_err_max": r["amp_err_max"],
                     "phase_err_deg_max": r["phase_err_deg_max"]})
    return {"series": errs}


def convergence_passes(series: list[dict[str, float]], amp_gate: float, phase_gate: float) -> bool:
    """Require improvement until the fixture reaches its accepted truncation floor."""

    if len(series) < 2:
        return False
    samples = [entry["samples_per_wavelength"] for entry in series]
    if any(current <= previous for previous, current in zip(samples[:-1], samples[1:], strict=True)):
        return False
    for key, gate in (("amp_err_max", amp_gate), ("phase_err_deg_max", phase_gate)):
        values = [entry[key] for entry in series]
        for previous, current in zip(values[:-1], values[1:], strict=True):
            if current > previous and not (previous < gate and current < gate):
                return False
        if values[-1] >= gate:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="P4-4 Kirchhoff kernel manufactured-fixture verification.")
    ap.add_argument("--config", type=Path, default=FIXTURE_CONFIG)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    gates = cfg["gates"]
    amp_gate = float(gates["amplitude_rel_max"]); ph_gate = float(gates["phase_deg_max"])

    cyl = run_cylindrical(cfg)
    pw = run_plane_wave(cfg)
    ctr = run_counterexample(cfg)
    conv = run_convergence(cfg)
    print("P4-4 Kirchhoff kernel fixtures (gates: amp <2%, phase <2 deg)")
    print(f"  1 cylindrical:   amp_err_max = {cyl['amp_err_max']:.5f}  "
          f"phase_err_max = {cyl['phase_err_deg_max']:.3f} deg")
    print("  2 convergence:   " + "  ".join(
        f"{e['samples_per_wavelength']:g}/lam:{e['amp_err_max']:.4f}" for e in conv["series"]))
    print(f"  3 plane/velocity: amp_err = {pw['amp_err']:.5f}  phase_err = {pw['phase_err_deg']:.3f} deg")
    print(f"  4 counterexample (hankel1 under exp(+iOmega t)): reconstruction error = "
          f"{ctr['reconstruction_error']:.3f} (must be > {gates['counterexample_min_error']})")
    ok = (cyl["amp_err_max"] < amp_gate and cyl["phase_err_deg_max"] < ph_gate
          and pw["amp_err"] < amp_gate and pw["phase_err_deg"] < ph_gate
          and convergence_passes(conv["series"], amp_gate, ph_gate)
          and ctr["reconstruction_error"] > float(gates["counterexample_min_error"]))
    print(f"\nK0 kernel gate: {'PASSED' if ok else 'FAILED'}")
    if args.out:
        args.out.write_text(json.dumps(
            {"cylindrical": cyl, "plane_wave": pw, "counterexample": ctr,
             "convergence": conv, "k0_passed": ok}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
